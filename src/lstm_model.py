"""
Part C – LSTM: predict next purchase channel (In-Store / Online)
from a sequence of recent purchases.

Usage (standalone):
    python src/lstm_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 4
FEATURE_COLS = ["channel", "time_slot", "amount", "is_weekend"]
TARGET_COL = "channel"  # next channel to predict


def load_and_clean(path: str) -> pd.DataFrame:
    """Load and clean the dirty transactions CSV."""
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    df = df.sort_values("transaction_date").reset_index(drop=True)

    # Drop rows where the target (channel) is missing
    df = df.dropna(subset=["channel"])

    # Fill missing time_slot with mode
    df["time_slot"] = df["time_slot"].fillna(df["time_slot"].mode()[0])

    # Fill missing amount with median
    df["amount"] = df["amount"].fillna(df["amount"].median())

    return df


def encode_features(df: pd.DataFrame):
    """Encode categorical features; return encoded df and encoders."""
    encoders = {}
    df = df.copy()
    for col in ["channel", "time_slot"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Normalise amount to [0, 1]
    df["amount"] = (df["amount"] - df["amount"].min()) / (
        df["amount"].max() - df["amount"].min()
    )
    return df, encoders


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH):
    """
    Create (X, y) where X[i] is the feature matrix of seq_len consecutive
    rows and y[i] is the channel of the row immediately after that window.
    """
    feature_vals = df[FEATURE_COLS].values.astype(np.float32)
    target_vals = df[TARGET_COL].values

    X, y = [], []
    for i in range(len(feature_vals) - seq_len):
        X.append(feature_vals[i : i + seq_len])
        y.append(target_vals[i + seq_len])

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_lstm_model(seq_len: int, n_features: int, n_classes: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(64, input_shape=(seq_len, n_features), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_lstm(path: str = "data/customer_transactions_dirty.csv"):
    """End-to-end pipeline: load → clean → encode → sequences → train → evaluate."""
    df = load_and_clean(path)
    df, encoders = encode_features(df)

    X, y = build_sequences(df, seq_len=SEQUENCE_LENGTH)

    n_classes = len(encoders[TARGET_COL].classes_)
    y_cat = to_categorical(y, num_classes=n_classes)

    # Chronological split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_cat[:split], y_cat[split:]
    y_test_labels = y[split:]

    model = build_lstm_model(
        seq_len=SEQUENCE_LENGTH,
        n_features=X.shape[2],
        n_classes=n_classes,
    )
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        verbose=1,
    )

    # Evaluate
    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test_labels, y_pred_labels)

    channel_labels = encoders[TARGET_COL].classes_
    print(f"\n=== LSTM Results ===")
    print(f"Training sequences : {len(X_train)}")
    print(f"Testing sequences  : {len(X_test)}")
    print(f"Accuracy           : {acc:.4f} ({acc*100:.2f}%)")

    # Confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=channel_labels, columns=channel_labels).to_string())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=channel_labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("LSTM – Confusion Matrix (next channel)")
    plt.tight_layout()
    plt.savefig("outputs/lstm_confusion_matrix.png", dpi=120)
    plt.close()
    print("Confusion matrix saved → outputs/lstm_confusion_matrix.png")

    # Training history plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("LSTM – Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("LSTM – Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/lstm_training_history.png", dpi=120)
    plt.close()
    print("Training history saved → outputs/lstm_training_history.png")

    return model, encoders, acc


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    train_lstm()
