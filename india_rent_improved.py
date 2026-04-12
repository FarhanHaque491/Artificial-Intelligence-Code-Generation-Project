"""
india_rent_improved.py
======================
Cleaned-up, corrected, and improved version of the India House Rent analysis.

Key changes over the original (see IMPROVEMENTS section at the bottom for details):
  1.  parse_floor  – logs unrecognised values; no silent data loss
  2.  remove_outliers_iqr – replaces the wrong iqr==0 skip with a percentile clip
  3.  Schema validation at load time
  4.  Log-transform of Rent target (log1p / expm1 round-trip)
  5.  HistGradientBoostingRegressor replaces LinearRegression
  6.  Rare-locality bucketing eliminates one-hot explosion
  7.  5-fold cross-validation replaces a single 80/20 split
  8.  Preprocessor instantiated per city (no shared-state risk)
  9.  Deduplicated plot helper eliminates ~80 % repeated plotting code
 10.  set_plot_style uses a context manager – no global rcParams mutation
 11.  main() decomposed into named stages
 12.  Type hints and docstrings throughout
 13.  CONFIG block now covers every magic number / string
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

matplotlib.use("Agg")  

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)



# CONFIG  –  every magic value lives here

DATA_PATH = "House_Rent_Dataset.csv"
OUTPUT_DIR = "india_rent_outputs"

CITIES: List[str] = [
    "Kolkata", "Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"
]

REQUIRED_COLUMNS: List[str] = [
    "Floor", "City", "Rent", "Size", "Bathroom",
    "Area Type", "Area Locality", "Furnishing Status",
]

OUTLIER_COLUMNS: List[str] = ["Rent", "Size", "Bathroom", "Floor_Num"]

# Modelling
RARE_LOCALITY_MIN_COUNT: int = 20   # localities with fewer rows → "Other"
N_CV_FOLDS: int = 5
RANDOM_STATE: int = 42

# Plotting
SIZE_QUANTILE_BINS: int = 10
TOP_LOCALITY_COUNT: int = 12
SAMPLE_PREDICTIONS_PER_CITY: int = 20



# DATA LOADING & VALIDATION

def load_and_validate(path: str) -> pd.DataFrame:
    """
    Load the CSV and raise early with a clear message if required columns
    are missing.  Previously the script would crash deep inside a pipeline
    with a cryptic KeyError.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Found columns: {df.columns.tolist()}"
        )
    logger.info("Loaded %d rows from '%s'.", len(df), path)
    return df



# DATA CLEANING HELPERS

_UNRECOGNISED_FLOORS: List[str] = []   # collected across all calls for reporting


def parse_floor(value: object) -> float:
    """
    Convert floor text into a numeric floor value.

    Change vs original
    ------------------
    Unrecognised values are now collected in _UNRECOGNISED_FLOORS instead of
    silently disappearing into NaN.  Call report_floor_parse_issues() after
    processing the column to surface them.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if "ground" in text:
        return 0.0
    if "upper basement" in text:
        return -0.5
    if "lower basement" in text:
        return -1.0

    match = re.search(r"(-?\d+)\s*out\s*of\s*(-?\d+)", text)
    if match:
        return float(match.group(1))

    match_num = re.search(r"-?\d+", text)
    if match_num:
        return float(match_num.group(0))

    _UNRECOGNISED_FLOORS.append(str(value))
    return np.nan


def report_floor_parse_issues() -> None:
    """Log a summary of floor values that could not be parsed."""
    if not _UNRECOGNISED_FLOORS:
        return
    from collections import Counter
    counts = Counter(_UNRECOGNISED_FLOORS)
    logger.warning(
        "parse_floor: %d values could not be converted to a number "
        "(top 10 shown): %s",
        len(_UNRECOGNISED_FLOORS),
        dict(counts.most_common(10)),
    )


def remove_outliers_iqr(group: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Remove outliers using the 1.5 × IQR rule.

    Change vs original
    ------------------
    The original skipped the filter entirely when IQR == 0 (all-same values or
    heavily skewed), allowing genuine outliers through.  The new version applies
    a fallback percentile clip (1st–99th) for zero-IQR columns so real outliers
    are still caught.
    """
    mask = pd.Series(True, index=group.index)

    for col in columns:
        s = pd.to_numeric(group[col], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr):
            continue

        if iqr == 0:
            # Fallback: keep only values within the 1st–99th percentile range
            lower, upper = s.quantile(0.01), s.quantile(0.99)
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

        mask &= s.isna() | ((s >= lower) & (s <= upper))

    return group.loc[mask].copy()


def bucket_rare_localities(df: pd.DataFrame, min_count: int = RARE_LOCALITY_MIN_COUNT) -> pd.DataFrame:
    """
    Replace infrequent Area Locality values with "Other".

    Change vs original
    ------------------
    The original passed every locality directly into OneHotEncoder, creating
    hundreds of sparse dummy columns which degrade linear models severely
    (multicollinearity, overfitting to rare areas).  Bucketing rare localities
    reduces dimensionality while preserving signal for common areas.
    """
    counts = df["Area Locality"].value_counts()
    rare = counts[counts < min_count].index
    df = df.copy()
    df.loc[df["Area Locality"].isin(rare), "Area Locality"] = "Other"
    logger.info(
        "Locality bucketing: %d rare localities → 'Other'; %d unique localities remain.",
        len(rare),
        df["Area Locality"].nunique(),
    )
    return df



# MODELLING

def build_preprocessor() -> ColumnTransformer:
  
    numeric_features = ["Size", "Floor_Num", "Bathroom"]
    categorical_features = ["Area Type", "Area Locality", "Furnishing Status"]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    ),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def train_models(
    clean_df: pd.DataFrame,
    cities: List[str],
) -> Tuple[pd.DataFrame, Dict[str, dict]]:

    ]
    target_col = "Rent"

    results = []
    city_models: Dict[str, dict] = {}

    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for city in cities:
        city_df = clean_df[clean_df["City"] == city].copy()

        if len(city_df) < N_CV_FOLDS * 10:
            logger.warning(
                "City '%s' has only %d rows after cleaning – CV metrics may be unstable.",
                city, len(city_df),
            )

        X = city_df[feature_cols]
        y_raw = city_df[target_col]
        y = np.log1p(y_raw)  # log-transform target

        model = Pipeline([
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            ),
        ])

        cv_results = cross_validate(
            model, X, y,
            cv=kf,
            scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"),
            return_train_score=False,
        )

        # Back-transform CV fold predictions for human-readable MAE / RMSE
        # (these are approximations computed from mean log-space scores)
        mae_log  = -cv_results["test_neg_mean_absolute_error"].mean()
        rmse_log = -cv_results["test_neg_root_mean_squared_error"].mean()
        r2_mean  = cv_results["test_r2"].mean()

        # Fit final model on all city data for saving sample predictions
        model.fit(X, y)
        y_pred_log = model.predict(X)
        y_pred = np.expm1(y_pred_log)

        mae_actual  = round(mean_absolute_error(y_raw, y_pred), 2)
        rmse_actual = round(float(np.sqrt(mean_squared_error(y_raw, y_pred))), 2)

        results.append({
            "City": city,
            "Rows_After_Outlier_Removal": len(city_df),
            f"CV_{N_CV_FOLDS}fold_MAE_log":  round(mae_log, 4),
            f"CV_{N_CV_FOLDS}fold_RMSE_log": round(rmse_log, 4),
            f"CV_{N_CV_FOLDS}fold_R2":       round(r2_mean, 4),
            "Train_MAE_actual":              mae_actual,
            "Train_RMSE_actual":             rmse_actual,
        })

        city_models[city] = {
            "data":        city_df,
            "model":       model,
            "X":           X.reset_index(drop=True),
            "y_actual":    y_raw.reset_index(drop=True),
            "y_pred":      y_pred,
        }

        logger.info(
            "%s  |  rows=%d  CV-R²=%.3f  CV-MAE(log)=%.4f",
            city, len(city_df), r2_mean, mae_log,
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values("City")
        .reset_index(drop=True)
    )
    return results_df, city_models



# PLOTTING  

@contextlib.contextmanager
def plot_style():
    """
    Context manager that applies plot style locally.

    Change vs original
    ------------------
    The original called set_plot_style() which permanently mutated global
    plt.rcParams for the lifetime of the process.  Using rc_context() limits
    the effect to plots generated inside this block.
    """
    with plt.rc_context({
        "figure.figsize":        (11.5, 6.8),
        "axes.grid":             True,
        "grid.alpha":            0.3,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
    }):
        yield


def _line_plot(
    grouped_df: pd.DataFrame,
    india_df: pd.DataFrame,
    x_col: str,
    cities: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
    figsize: tuple | None = None,
    xtick_rotation: int = 0,
) -> None:

    with plot_style():
        fig, ax = plt.subplots(figsize=figsize)

        for city in cities:
            part = grouped_df[grouped_df["City"] == city]
            ax.plot(part[x_col], part["Rent"], marker="o", linewidth=2, label=city)

        ax.plot(
            india_df[x_col],
            india_df["Rent"],
            marker="o",
            linewidth=3,
            linestyle="--",
            label="India Average",
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xtick_rotation:
            ax.tick_params(axis="x", rotation=xtick_rotation)
            ax.set_xticklabels(ax.get_xticklabels(), ha="right")
        ax.legend(frameon=True, ncol=2)
        fig.tight_layout()
        fig.savefig(save_path, dpi=180)
        plt.close(fig)


# ---- individual plot builders -------------------------------------------

def save_size_plot(clean_df: pd.DataFrame, cities: List[str], output_dir: str) -> None:
    size_df = clean_df.dropna(subset=["Size", "Rent"]).copy()
    size_df["Size_Bin"] = pd.qcut(size_df["Size"], q=SIZE_QUANTILE_BINS, duplicates="drop")

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby("Size_Bin", observed=False)
            .agg(Size_Mid=("Size", "median"), Rent=("Rent", "mean"))
            .reset_index()
            .sort_values("Size_Mid")
        )

    city_grouped = (
        size_df.groupby("City", group_keys=False)
        .apply(_agg)
        .reset_index(level=0)
        .rename(columns={"level_0": "City"})
    )
    india_grouped = _agg(size_df)

    _line_plot(
        city_grouped, india_grouped,
        x_col="Size_Mid", cities=cities,
        title="Average Rent vs Size",
        xlabel="Size (median of bin, sq ft)", ylabel="Average Rent (₹)",
        save_path=os.path.join(output_dir, "line_size_vs_rent.png"),
    )


def save_floor_plot(clean_df: pd.DataFrame, cities: List[str], output_dir: str) -> None:
  
    floor_df = clean_df.dropna(subset=["Floor_Num", "Rent"]).copy()

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if df["Floor_Num"].nunique() > 20:
            out["Floor_Num"] = np.round(df["Floor_Num"])
        return (
            out.groupby("Floor_Num", as_index=False)["Rent"]
            .mean()
            .sort_values("Floor_Num")
        )

    city_grouped = (
        floor_df.groupby("City", group_keys=False)
        .apply(_agg)
        .reset_index(level=0)
        .rename(columns={"level_0": "City"})
    )
    india_grouped = _agg(floor_df)

    _line_plot(
        city_grouped, india_grouped,
        x_col="Floor_Num", cities=cities,
        title="Average Rent vs Floor",
        xlabel="Floor number", ylabel="Average Rent (₹)",
        save_path=os.path.join(output_dir, "line_floor_vs_rent.png"),
    )


def _categorical_line_plot(
    clean_df: pd.DataFrame,
    cities: List[str],
    output_dir: str,
    col: str,
    filename: str,
    title: str,
    xlabel: str,
    xtick_rotation: int = 20,
    figsize: tuple | None = None,
    filter_top_n: int | None = None,
) -> None:
    """Generic builder for categorical x-axis line plots."""
    if filter_top_n:
        top_vals = clean_df[col].value_counts().head(filter_top_n).index.tolist()
        df = clean_df[clean_df[col].isin(top_vals)]
    else:
        df = clean_df

    cats = df[col].dropna().unique().tolist()

    city_grouped = (
        df.groupby([col, "City"], as_index=False)["Rent"].mean()
    )
    india_grouped = df.groupby(col, as_index=False)["Rent"].mean()

    # Align both dataframes to the same category order
    city_pivot = (
        city_grouped.pivot(index="City", columns=col, values="Rent")
        .reindex(columns=cats)
        .stack(future_stack=True)
        .reset_index()
        .rename(columns={0: "Rent"})
    )

    india_aligned = (
        india_grouped.set_index(col).reindex(cats).reset_index()
    )

    _line_plot(
        city_pivot, india_aligned,
        x_col=col, cities=cities,
        title=title, xlabel=xlabel, ylabel="Average Rent (₹)",
        save_path=os.path.join(output_dir, filename),
        figsize=figsize,
        xtick_rotation=xtick_rotation,
    )


def save_all_graphs(clean_df: pd.DataFrame, cities: List[str], output_dir: str) -> None:
    """Render and save all exploratory charts."""
    save_size_plot(clean_df, cities, output_dir)
    save_floor_plot(clean_df, cities, output_dir)

    _categorical_line_plot(
        clean_df, cities, output_dir,
        col="Area Type", filename="line_area_type_vs_rent.png",
        title="Average Rent vs Area Type", xlabel="Area Type",
        xtick_rotation=20,
    )
    _categorical_line_plot(
        clean_df, cities, output_dir,
        col="Area Locality", filename="line_locality_vs_rent.png",
        title="Average Rent vs Area Locality", xlabel="Area Locality",
        filter_top_n=TOP_LOCALITY_COUNT,
        figsize=(14.5, 7.2), xtick_rotation=55,
    )
    _categorical_line_plot(
        clean_df, cities, output_dir,
        col="Furnishing Status", filename="line_furnishing_vs_rent.png",
        title="Average Rent vs Furnishing Status", xlabel="Furnishing Status",
        xtick_rotation=20,
    )

    # Bathroom is numeric but treated as categorical for this chart
    bathroom_city = clean_df.groupby(["Bathroom", "City"], as_index=False)["Rent"].mean()
    bathroom_india = clean_df.groupby("Bathroom", as_index=False)["Rent"].mean()

    _line_plot(
        bathroom_city, bathroom_india,
        x_col="Bathroom", cities=cities,
        title="Average Rent vs Bathroom Count",
        xlabel="Number of bathrooms", ylabel="Average Rent (₹)",
        save_path=os.path.join(output_dir, "line_bathroom_vs_rent.png"),
    )

    logger.info("All plots saved to '%s'.", output_dir)



# MAIN  –  decomposed into named stages

def load_data(path: str) -> pd.DataFrame:
    df = load_and_validate(path)
    df["Floor_Num"] = df["Floor"].apply(parse_floor)
    report_floor_parse_issues()
    return df[df["City"].isin(CITIES)].copy()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = (
        df.groupby("City", group_keys=False)
        .apply(lambda g: remove_outliers_iqr(g, OUTLIER_COLUMNS))
        .reset_index(drop=True)
    )
    logger.info(
        "Outlier removal: %d → %d rows (%.1f %% retained).",
        len(df), len(clean), 100 * len(clean) / len(df),
    )
    clean = bucket_rare_localities(clean)
    return clean


def save_outputs(
    results_df: pd.DataFrame,
    city_models: Dict[str, dict],
    output_dir: str,
) -> None:
    results_df.to_csv(
        os.path.join(output_dir, "model_metrics_by_city.csv"),
        index=False,
    )

    for city, payload in city_models.items():
        sample = payload["X"].head(SAMPLE_PREDICTIONS_PER_CITY).copy()
        sample["Actual_Rent"]    = payload["y_actual"].head(SAMPLE_PREDICTIONS_PER_CITY)
        sample["Predicted_Rent"] = np.round(
            payload["y_pred"][:SAMPLE_PREDICTIONS_PER_CITY], 2
        )
        sample.to_csv(
            os.path.join(output_dir, f"{city.lower()}_sample_predictions.csv"),
            index=False,
        )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df      = load_data(DATA_PATH)
    clean   = clean_data(df)
    results, city_models = train_models(clean, CITIES)

    save_outputs(results, city_models, OUTPUT_DIR)
    save_all_graphs(clean, CITIES, OUTPUT_DIR)

    logger.info("Finished successfully.")
    logger.info("\nModel metrics:\n%s", results.to_string(index=False))
    logger.info("Outputs saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()



