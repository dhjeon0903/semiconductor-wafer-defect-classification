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


print("=== OPEN WAFER DATASET ===")

df = pd.read_pickle("wm811k_converted.pkl")

print("\n[Raw head]")
print(df.head())

print("\n[Columns]")
print(df.columns)

print("\n[Shape]")
print(df.shape)

df["trianTestLabel"] = df["trianTestLabel"].apply(unwrap_label)
df["failureType"] = df["failureType"].apply(unwrap_label)

print("\n[Train/Test counts]")
print(df["trianTestLabel"].value_counts(dropna=False))

print("\n[Failure type counts]")
print(df["failureType"].value_counts(dropna=False))

print("\n[Wafer map sample shape]")
print(np.array(df.iloc[0]["waferMap"]).shape)