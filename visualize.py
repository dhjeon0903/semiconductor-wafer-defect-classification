import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def load_data(filepath="wm811k_converted.pkl"):
    df = pd.read_pickle(filepath).copy()
    df["trianTestLabel"] = df["trianTestLabel"].apply(unwrap_label)
    df["failureType"] = df["failureType"].apply(unwrap_label)
    df["binaryLabel"] = df["failureType"].apply(lambda x: 0 if x == "none" else 1)
    return df


if __name__ == "__main__":
    print("=== VISUALIZING WAFER MAPS ===")

    df = load_data("wm811k.pkl")
    sample_indices = [0, 100, 1000, 5000]

    for idx in sample_indices:
        wafer_map = np.array(df.iloc[idx]["waferMap"])
        failure_type = df.iloc[idx]["failureType"]

        plt.figure(figsize=(5, 5))
        plt.imshow(wafer_map)
        plt.title(f"Index: {idx}, Failure: {failure_type}")
        plt.colorbar()
        plt.show()