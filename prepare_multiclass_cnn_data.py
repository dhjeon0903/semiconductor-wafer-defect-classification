import os
import pandas as pd
import numpy as np

TARGET_SIZE = (64, 64)
OUTPUT_DIR = "cnn_data_multi"

VALID_CLASSES = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]


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


def load_and_clean_data(filepath="wm811k_converted.pkl"):
    df = pd.read_pickle(filepath).copy()
    df["trianTestLabel"] = df["trianTestLabel"].apply(unwrap_label)
    df["failureType"] = df["failureType"].apply(unwrap_label)
    return df


def resize_nearest(arr, target_size=(64, 64)):
    arr = np.array(arr, dtype=np.float32)
    src_h, src_w = arr.shape
    tgt_h, tgt_w = target_size

    row_idx = np.linspace(0, src_h - 1, tgt_h).astype(int)
    col_idx = np.linspace(0, src_w - 1, tgt_w).astype(int)

    return arr[row_idx][:, col_idx]


def normalize_map(arr):
    arr = np.array(arr, dtype=np.float32)
    return arr / 2.0


def build_label_map(classes):
    return {label: idx for idx, label in enumerate(classes)}


def balanced_sample(df, per_class_limit):
    parts = []

    for cls in VALID_CLASSES:
        cls_df = df[df["failureType"] == cls].copy()
        if len(cls_df) == 0:
            continue

        n = min(len(cls_df), per_class_limit)
        parts.append(cls_df.sample(n=n, random_state=42))

    sampled_df = pd.concat(parts, axis=0).reset_index(drop=True)
    return sampled_df


def build_dataset(df, label_map, target_size=(64, 64)):
    X = []
    y = []

    for row in df.itertuples(index=False):
        label = row.failureType
        if label not in label_map:
            continue

        wafer_map = np.array(row.waferMap, dtype=np.float32)
        wafer_map = resize_nearest(wafer_map, target_size)
        wafer_map = normalize_map(wafer_map)
        wafer_map = np.expand_dims(wafer_map, axis=-1)

        X.append(wafer_map)
        y.append(label_map[label])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


if __name__ == "__main__":
    print("=== PREPARING IMPROVED MULTICLASS CNN DATA ===")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_clean_data("wm811k_converted.pkl")
    df = df[df["failureType"].isin(VALID_CLASSES)].copy()

    train_df = df[df["trianTestLabel"] == "Training"].copy()
    test_df = df[df["trianTestLabel"] == "Test"].copy()

    print("\n[Original train distribution]")
    print(train_df["failureType"].value_counts())

    print("\n[Original test distribution]")
    print(test_df["failureType"].value_counts())

    # improving point
    train_per_class = 1200
    test_per_class = 300

    train_df = balanced_sample(train_df, train_per_class)
    test_df = balanced_sample(test_df, test_per_class)

    print("\n[Balanced train distribution]")
    print(train_df["failureType"].value_counts())

    print("\n[Balanced test distribution]")
    print(test_df["failureType"].value_counts())

    label_map = build_label_map(VALID_CLASSES)

    X_train, y_train = build_dataset(train_df, label_map, TARGET_SIZE)
    X_test, y_test = build_dataset(test_df, label_map, TARGET_SIZE)

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(OUTPUT_DIR, "label_map.npy"), label_map, allow_pickle=True)

    print("\nSaved files:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)