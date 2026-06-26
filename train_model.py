"""
train_model.py
----------------
STEP 2: Neural network training.

Loads your 6 gesture CSVs from dataset/, preprocesses them, trains a small
MLP (multi-layer perceptron) classifier, and saves:
    model/gesture_model.h5        <- trained Keras model
    model/label_map.json          <- {0: "Closed Hand", 1: "V Sign", ...}
    model/training_history.png    <- accuracy / loss curves
    model/confusion_matrix.png    <- evaluation on held-out test set

Run this from PyCharm (or `python train_model.py` in the project folder)
after placing your 6 CSVs inside the dataset/ folder, named exactly:
    closed_hand.csv, v_sign.csv, thumb_out.csv,
    opened_hand.csv, index_finger.csv, three_fingers.csv

If your filenames differ, either rename them or edit GESTURES in config.py.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from config import (
    GESTURES, NUM_CLASSES, DATASET_DIR, MODEL_PATH,
    LABEL_MAP_PATH, HISTORY_PLOT_PATH, CONFUSION_MATRIX_PATH,
)
from preprocessing import normalize_landmarks, FEATURE_LENGTH

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Gesture_Recognition")

print("Tracking URI:", mlflow.get_tracking_uri())

experiment = mlflow.get_experiment_by_name(
    "Gesture_Recognition"
)

print("Experiment:", experiment)


def load_dataset():
    """
    Load every gesture CSV, remove the label column,
    normalize landmarks, and build X/y arrays.
    """
    X, y = [], []

    for gesture_id, info in GESTURES.items():
        csv_path = f"{DATASET_DIR}/{info['file']}.csv"

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find {csv_path}"
            )

        if "label" in df.columns:
            df = df.drop(columns=["label"])

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

        print(f"Checking {csv_path}")
        print(f"Shape after cleanup: {df.shape}")

        if df.shape[1] != FEATURE_LENGTH:
            raise ValueError(
                f"{csv_path} has {df.shape[1]} columns, expected {FEATURE_LENGTH} columns."
            )

        print(
            f"[{gesture_id}] {info['label']:<15} -> "
            f"{len(df)} samples"
        )

        for row in df.values:
            X.append(normalize_landmarks(row))
            y.append(gesture_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


def build_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.close(fig)
    print(f"Saved training curves -> {HISTORY_PLOT_PATH}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (test set)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)
    print(f"Saved confusion matrix -> {CONFUSION_MATRIX_PATH}")


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"\nTotal samples: {len(X)}   Feature length: {X.shape[1]}")

    class_names = [GESTURES[i]["label"] for i in range(NUM_CLASSES)]

    # Stratified split keeps gesture proportions equal across train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    y_train_oh = to_categorical(y_train, NUM_CLASSES)
    y_val_oh = to_categorical(y_val, NUM_CLASSES)

    model = build_model(input_dim=X.shape[1], num_classes=NUM_CLASSES)
    model.summary()

    with mlflow.start_run():
        # Log model/training settings
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 150)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("dropout1", 0.3)
        mlflow.log_param("dropout2", 0.3)
        mlflow.log_param("dropout3", 0.2)
        mlflow.log_param("optimizer", "Adam")

        # Dataset information
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("feature_length", FEATURE_LENGTH)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("validation_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ]

        history = model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=2,
        )

        # Log best training metrics
        mlflow.log_metric(
            "best_val_accuracy",
            max(history.history["val_accuracy"])
        )

        mlflow.log_metric(
            "best_val_loss",
            min(history.history["val_loss"])
        )

        mlflow.log_metric(
            "final_train_accuracy",
            history.history["accuracy"][-1]
        )

        mlflow.log_metric(
            "final_train_loss",
            history.history["loss"][-1]
        )

        plot_history(history)

        # --- final evaluation on the untouched test set ---
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print("\nClassification report (test set):")
        print(classification_report(y_test, y_pred, target_names=class_names))

        test_accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        plot_confusion_matrix(y_test, y_pred, class_names)

        # model.h5 was already saved by ModelCheckpoint (best val_accuracy epoch),
        # but save again at the end in case checkpointing was skipped on a tiny dataset.
        model.save(MODEL_PATH)
        print(f"\nSaved trained model -> {MODEL_PATH}")

        mlflow.tensorflow.log_model(
            model,
            artifact_path="gesture_model"
        )

        label_map = {str(i): GESTURES[i]["label"] for i in range(NUM_CLASSES)}
        with open(LABEL_MAP_PATH, "w") as f:
            json.dump(label_map, f, indent=2)
        print(f"Saved label map -> {LABEL_MAP_PATH}")

        # Save generated plots/files
        mlflow.log_artifact(HISTORY_PLOT_PATH)
        mlflow.log_artifact(CONFUSION_MATRIX_PATH)
        mlflow.log_artifact(LABEL_MAP_PATH)

        print("\nDone. Next step: run realtime_classifier.py to test the model live on your webcam.")


if __name__ == "__main__":
    main()
