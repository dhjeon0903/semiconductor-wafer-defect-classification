import pandas as pd
import numpy as np


def unwrap_label(x):
    if isinstance(x, str):
        return x
    if x is None:
        return "unknown"

    try:
        arr = np.array(x, dtype=object)
        if arr.size == 0:
            return "unknown"

        while isinstance(arr, np.ndarray) and arr.size > 0:
            arr = arr[0]

        if arr is None:
            return "unknown"

        return str(arr)
    except Exception:
        return "unknown"


def load_and_clean_data(filepath="wm811k.pkl"):
    df = pd.read_pickle(filepath).copy()

    df["trianTestLabel"] = df["trianTestLabel"].apply(unwrap_label)
    df["failureType"] = df["failureType"].apply(unwrap_label)
    df["binaryLabel"] = df["failureType"].apply(lambda x: 0 if x == "none" else 1)

    return df


def extract_basic_features(wafer_map):
    wafer_map = np.array(wafer_map)

    rows, cols = wafer_map.shape
    total_cells = wafer_map.size
    valid_cells = np.sum(wafer_map > 0)
    normal_cells = np.sum(wafer_map == 1)
    defect_cells = np.sum(wafer_map == 2)

    normal_ratio = normal_cells / valid_cells if valid_cells > 0 else 0.0
    defect_ratio = defect_cells / valid_cells if valid_cells > 0 else 0.0

    return {
        "rows": rows,
        "cols": cols,
        "total_cells": int(total_cells),
        "valid_cells": int(valid_cells),
        "normal_cells": int(normal_cells),
        "defect_cells": int(defect_cells),
        "normal_ratio": float(normal_ratio),
        "defect_ratio": float(defect_ratio),
    }


def build_feature_dataframe(df):
    feature_rows = []

    for _, row in df.iterrows():
        features = extract_basic_features(row["waferMap"])
        features["trianTestLabel"] = row["trianTestLabel"]
        features["failureType"] = row["failureType"]
        features["binaryLabel"] = row["binaryLabel"]
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    print("=== PREPROCESSING DATA ===")

    df = load_and_clean_data("wm811k_converted.pkl")

    print("\n[Original shape]")
    print(df.shape)

    print("\n[Train/Test counts]")
    print(df["trianTestLabel"].value_counts(dropna=False))

    print("\n[Failure type counts]")
    print(df["failureType"].value_counts(dropna=False))

    feature_df = build_feature_dataframe(df)

    print("\n[Feature dataframe shape]")
    print(feature_df.shape)

    print("\n[Feature dataframe head]")
    print(feature_df.head())

    feature_df.to_csv("wafer_features.csv", index=False)
    print("\nSaved: wafer_features.csv")