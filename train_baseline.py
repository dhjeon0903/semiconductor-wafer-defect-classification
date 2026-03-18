import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURE_COLUMNS = [
    "rows",
    "cols",
    "total_cells",
    "valid_cells",
    "normal_cells",
    "defect_cells",
    "normal_ratio",
    "defect_ratio",
]


def load_feature_data(filepath="wafer_features.csv"):
    return pd.read_csv(filepath)


if __name__ == "__main__":
    print("=== TRAINING BASELINE MODEL ===")

    df = load_feature_data("wafer_features.csv")

    train_df = df[df["trianTestLabel"] == "Training"].copy()
    test_df = df[df["trianTestLabel"] == "Test"].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["binaryLabel"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["binaryLabel"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n[Accuracy]")
    print(accuracy_score(y_test, y_pred))

    print("\n[Confusion Matrix]")
    print(confusion_matrix(y_test, y_pred))

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred))