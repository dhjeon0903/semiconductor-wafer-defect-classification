import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = "cnn_data_multi"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "wafer_cnn_multiclass_improved.keras")


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    label_map = np.load(os.path.join(DATA_DIR, "label_map.npy"), allow_pickle=True).item()
    return X_train, y_train, X_test, y_test, label_map


def build_model(input_shape, num_classes):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
    ])

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        data_augmentation,

        layers.Conv2D(
            32, (3, 3), activation="relu", padding="same",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.20),

        layers.Conv2D(
            64, (3, 3), activation="relu", padding="same",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(
            128, (3, 3), activation="relu", padding="same",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        layers.Flatten(),

        layers.Dense(
            128, activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.Dropout(0.40),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_training_history(history, output_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_sample_predictions(X_test, y_true, y_pred, class_names, output_path, n=12):
    indices = list(range(len(X_test)))
    random.seed(42)
    chosen = random.sample(indices, min(n, len(indices)))

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(chosen, start=1):
        plt.subplot(3, 4, i)
        plt.imshow(X_test[idx].squeeze(), cmap="viridis")
        plt.title(f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=== TRAINING IMPROVED MULTICLASS CNN MODEL ===")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test, label_map = load_data()

    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    print("\n[X_train shape]")
    print(X_train.shape)

    print("\n[X_test shape]")
    print(X_test.shape)

    print("\n[Classes]")
    print(class_names)

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print("\n[Test Loss]")
    print(test_loss)

    print("\n[Test Accuracy]")
    print(test_acc)

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))

    save_training_history(history, os.path.join(OUTPUT_DIR, "training_history_improved.png"))
    save_confusion_matrix(y_test, y_pred, class_names, os.path.join(OUTPUT_DIR, "confusion_matrix_improved.png"))
    save_sample_predictions(X_test, y_test, y_pred, class_names, os.path.join(OUTPUT_DIR, "sample_predictions_improved.png"))

    model.save(MODEL_PATH)

    print("\nSaved outputs:")
    print(os.path.join(OUTPUT_DIR, "training_history_improved.png"))
    print(os.path.join(OUTPUT_DIR, "confusion_matrix_improved.png"))
    print(os.path.join(OUTPUT_DIR, "sample_predictions_improved.png"))
    print(MODEL_PATH)