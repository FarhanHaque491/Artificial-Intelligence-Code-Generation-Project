
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "House_Rent_Dataset.csv"
OUTPUT_DIR = "india_rent_outputs"

CITIES = ["Kolkata", "Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"]
OUTLIER_COLUMNS = ["Rent", "Size", "Bathroom", "Floor_Num"]


# ============================================================
# DATA CLEANING HELPERS
# ============================================================
def parse_floor(value):
    """Convert floor text into a numeric floor value."""
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

    return np.nan


def remove_outliers_iqr(group, columns):
    """Remove outliers from a grouped dataframe using the 1.5 * IQR rule."""
    mask = pd.Series(True, index=group.index)

    for col in columns:
        s = pd.to_numeric(group[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask &= s.isna() | ((s >= lower) & (s <= upper))

    return group.loc[mask].copy()


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================
# MODELING
# ============================================================
def build_preprocessor():
    numeric_features = ["Size", "Floor_Num", "Bathroom"]
    categorical_features = ["Area Type", "Area Locality", "Furnishing Status"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features,
            ),
        ]
    )
    return preprocessor


def train_models(clean_df, cities):
    feature_cols = [
        "Size",
        "Floor_Num",
        "Area Type",
        "Area Locality",
        "Furnishing Status",
        "Bathroom",
    ]
    target_col = "Rent"

    preprocessor = build_preprocessor()

    results = []
    city_models = {}

    for city in cities:
        city_df = clean_df[clean_df["City"] == city].copy()

        X = city_df[feature_cols]
        y = city_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "City": city,
            "Rows_After_Outlier_Removal": len(city_df),
            "MAE": round(mean_absolute_error(y_test, predictions), 2),
            "RMSE": round(rmse(y_test, predictions), 2),
            "R2": round(r2_score(y_test, predictions), 4),
        })

        city_models[city] = {
            "data": city_df,
            "model": model,
            "X_test": X_test.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "predictions": predictions,
        }

    results_df = pd.DataFrame(results).sort_values("City").reset_index(drop=True)
    return results_df, city_models


# ============================================================
# PLOTTING
# ============================================================
def set_plot_style():
    plt.rcParams["figure.figsize"] = (11.5, 6.8)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def save_size_plot(clean_df, cities, output_dir):
    size_df = clean_df.dropna(subset=["Size", "Rent"]).copy()
    size_df["Size_Bin"] = pd.qcut(size_df["Size"], q=10, duplicates="drop")

    size_grouped = (
        size_df.groupby(["City", "Size_Bin"], observed=False)
        .agg(Size_Mid=("Size", "median"), Avg_Rent=("Rent", "mean"))
        .reset_index()
        .sort_values(["City", "Size_Mid"])
    )

    india_size_grouped = (
        size_df.groupby("Size_Bin", observed=False)
        .agg(Size_Mid=("Size", "median"), Avg_Rent=("Rent", "mean"))
        .reset_index()
        .sort_values("Size_Mid")
    )

    plt.figure()
    for city in cities:
        part = size_grouped[size_grouped["City"] == city]
        plt.plot(part["Size_Mid"], part["Avg_Rent"], marker="o", linewidth=2, label=city)

    plt.plot(
        india_size_grouped["Size_Mid"],
        india_size_grouped["Avg_Rent"],
        marker="o",
        linewidth=3,
        linestyle="--",
        label="India Average"
    )

    plt.title("Average Rent vs Size")
    plt.xlabel("Size (median of bin, sq ft)")
    plt.ylabel("Average Rent")
    plt.legend(frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "line_size_vs_rent_with_india_average.png"), dpi=180)
    plt.close()


def save_floor_plot(clean_df, cities, output_dir):
    floor_grouped = (
        clean_df.dropna(subset=["Floor_Num", "Rent"])
        .groupby(["City", "Floor_Num"], as_index=False)["Rent"]
        .mean()
        .sort_values(["City", "Floor_Num"])
    )

    india_floor_grouped = (
        clean_df.dropna(subset=["Floor_Num", "Rent"])
        .groupby("Floor_Num", as_index=False)["Rent"]
        .mean()
        .sort_values("Floor_Num")
    )

    plt.figure()

    for city in cities:
        part = floor_grouped[floor_grouped["City"] == city]

        if len(part) > 20:
            part = (
                part.groupby(np.round(part["Floor_Num"]))
                .agg({"Rent": "mean"})
                .reset_index()
                .rename(columns={"Floor_Num": "Floor"})
            )
            xvals = part["Floor"]
            yvals = part["Rent"]
        else:
            xvals = part["Floor_Num"]
            yvals = part["Rent"]

        plt.plot(xvals, yvals, marker="o", linewidth=2, label=city)

    india_part = india_floor_grouped.copy()
    if len(india_part) > 20:
        india_part = (
            india_part.groupby(np.round(india_part["Floor_Num"]))
            .agg({"Rent": "mean"})
            .reset_index()
            .rename(columns={"Floor_Num": "Floor"})
        )
        xvals = india_part["Floor"]
        yvals = india_part["Rent"]
    else:
        xvals = india_part["Floor_Num"]
        yvals = india_part["Rent"]

    plt.plot(xvals, yvals, marker="o", linewidth=3, linestyle="--", label="India Average")

    plt.title("Average Rent vs Floor")
    plt.xlabel("Floor number")
    plt.ylabel("Average Rent")
    plt.legend(frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "line_floor_vs_rent_with_india_average.png"), dpi=180)
    plt.close()


def save_area_type_plot(clean_df, cities, output_dir):
    area_type_grouped = clean_df.groupby(["Area Type", "City"], as_index=False)["Rent"].mean()
    india_area_type_grouped = clean_df.groupby("Area Type", as_index=False)["Rent"].mean()

    area_types = [x for x in area_type_grouped["Area Type"].dropna().unique()]

    plt.figure()

    for city in cities:
        part = (
            area_type_grouped[area_type_grouped["City"] == city]
            .set_index("Area Type")
            .reindex(area_types)
            .reset_index()
        )
        plt.plot(part["Area Type"], part["Rent"], marker="o", linewidth=2, label=city)

    india_part = india_area_type_grouped.set_index("Area Type").reindex(area_types).reset_index()
    plt.plot(
        india_part["Area Type"],
        india_part["Rent"],
        marker="o",
        linewidth=3,
        linestyle="--",
        label="India Average"
    )

    plt.title("Average Rent vs Area Type")
    plt.xlabel("Area Type")
    plt.ylabel("Average Rent")
    plt.xticks(rotation=20)
    plt.legend(frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "line_area_type_vs_rent_with_india_average.png"), dpi=180)
    plt.close()


def save_locality_plot(clean_df, cities, output_dir):
    top_localities = clean_df["Area Locality"].value_counts().head(12).index.tolist()

    locality_grouped = (
        clean_df[clean_df["Area Locality"].isin(top_localities)]
        .groupby(["Area Locality", "City"], as_index=False)["Rent"]
        .mean()
    )

    india_locality_grouped = (
        clean_df[clean_df["Area Locality"].isin(top_localities)]
        .groupby("Area Locality", as_index=False)["Rent"]
        .mean()
    )

    plt.figure(figsize=(14.5, 7.2))

    for city in cities:
        part = (
            locality_grouped[locality_grouped["City"] == city]
            .set_index("Area Locality")
            .reindex(top_localities)
            .reset_index()
        )
        plt.plot(part["Area Locality"], part["Rent"], marker="o", linewidth=2, label=city)

    india_part = india_locality_grouped.set_index("Area Locality").reindex(top_localities).reset_index()
    plt.plot(
        india_part["Area Locality"],
        india_part["Rent"],
        marker="o",
        linewidth=3,
        linestyle="--",
        label="India Average"
    )

    plt.title("Average Rent vs Area Locality")
    plt.xlabel("Area Locality")
    plt.ylabel("Average Rent")
    plt.xticks(rotation=55, ha="right")
    plt.legend(frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "line_locality_vs_rent_with_india_average.png"), dpi=180)
    plt.close()


def save_furnishing_plot(clean_df, cities, output_dir):
    furnishing_grouped = clean_df.groupby(["Furnishing Status", "City"], as_index=False)["Rent"].mean()
    india_furnishing_grouped = clean_df.groupby("Furnishing Status", as_index=False)["Rent"].mean()

    statuses = [x for x in furnishing_grouped["Furnishing Status"].dropna().unique()]

    plt.figure()

    for city in cities:
        part = (
            furnishing_grouped[furnishing_grouped["City"] == city]
            .set_index("Furnishing Status")
            .reindex(statuses)
            .reset_index()
        )
        plt.plot(part["Furnishing Status"], part["Rent"], marker="o", linewidth=2, label=city)

    india_part = india_furnishing_grouped.set_index("Furnishing Status").reindex(statuses).reset_index()
    plt.plot(
        india_part["Furnishing Status"],
        india_part["Rent"],
        marker="o",
        linewidth=3,
        linestyle="--",
        label="India Average"
    )

    plt.title("Average Rent vs Furnishing Status")
    plt.xlabel("Furnishing Status")
    plt.ylabel("Average Rent")
    plt.xticks(rotation=20)
    plt.legend(frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "line_furnishing_vs_rent_with_india_average.png"), dpi=180)
    plt.close()


def save_bathroom_plot(clean_df, cities, output_dir):
    bathroom_grouped = (
        clean_df.groupby(["Bathroom", "City"], as_index=False)["Rent"]
        .mean()
        .sort_values(["City", "Bathroom"])
    )

    india_bathroom_grouped = (
        clean_df.groupby("Bathroom", as_index=False)["Rent"]
        .mean()
        .sort_values("Bathroom")
    )

    plt.figure()

    for city in cities:
        part = bathroom_grouped[bathroom_grouped["City"] == city]
        plt.plot(part["Bathroom"], part["Rent"], marker="o", linewidth=2, label=city)

    plt.plot(
        india_bathroom_grouped["Bathroom"],
        india_bathroom_grouped["Rent"],
        marker="o",
        linewidth=3,
        linestyle="--",
        label="India Average"
    )

    plt.title("Average Rent vs Bathroom Count")
    plt.xlabel("Number of bathrooms")
    plt.ylabel("Average Rent")
    plt.legend(frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "line_bathroom_vs_rent_with_india_average.png"), dpi=180)
    plt.close()


def save_all_graphs(clean_df, cities, output_dir):
    set_plot_style()
    save_size_plot(clean_df, cities, output_dir)
    save_floor_plot(clean_df, cities, output_dir)
    save_area_type_plot(clean_df, cities, output_dir)
    save_locality_plot(clean_df, cities, output_dir)
    save_furnishing_plot(clean_df, cities, output_dir)
    save_bathroom_plot(clean_df, cities, output_dir)


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["Floor_Num"] = df["Floor"].apply(parse_floor)
    df = df[df["City"].isin(CITIES)].copy()

    clean_df = (
        df.groupby("City", group_keys=False)
        .apply(lambda g: remove_outliers_iqr(g, OUTLIER_COLUMNS))
        .reset_index(drop=True)
    )

    results_df, city_models = train_models(clean_df, CITIES)
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "outlier_adjusted_model_metrics_by_city.csv"),
        index=False
    )

    # Save sample predictions for one example city per folder-style use
    for city, payload in city_models.items():
        city_pred_path = os.path.join(
            OUTPUT_DIR,
            f"{city.lower()}_sample_predictions.csv"
        )
        sample_predictions = payload["X_test"].copy()
        sample_predictions["Actual_Rent"] = payload["y_test"]
        sample_predictions["Predicted_Rent"] = np.round(payload["predictions"], 2)
        sample_predictions.head(20).to_csv(city_pred_path, index=False)

    save_all_graphs(clean_df, CITIES, OUTPUT_DIR)

    print("Finished successfully.")
    print("\nModel metrics:")
    print(results_df.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
