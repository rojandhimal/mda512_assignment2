"""
Part B – Random Forest: predict time_slot from customer features.

Usage (standalone):
    python src/random_forest.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """Load the dirty CSV and apply basic cleaning."""
    df = pd.read_csv(path)

    # Drop rows where the target is missing
    df = df.dropna(subset=["time_slot"])

    # Numeric columns: fill missing with median
    for col in ["morning_ratio", "Age", "Amount"]:
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill missing with mode
    for col in ["Channel", "prev_slot", "Sex"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_features(df: pd.DataFrame):
    """Label-encode categorical predictors and the target."""
    cat_cols = ["prev_slot", "Channel", "Sex", "time_slot"]
    encoders = {}
    df = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

FEATURE_COLS = ["prev_slot", "morning_ratio", "is_weekend", "Channel", "Sex", "Age", "Amount"]
TARGET_COL = "time_slot"


def train_random_forest(path: str = "data/customer_model_table_dirty.csv"):
    """End-to-end pipeline: load → clean → split → train → evaluate."""
    df = load_and_clean(path)
    df, encoders = encode_features(df)

    # Chronological train/test split (earlier rows → train, later rows → test)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[FEATURE_COLS].values, train[TARGET_COL].values
    X_test, y_test = test[FEATURE_COLS].values, test[TARGET_COL].values

    # Fit model
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Random Forest Results ===")
    print(f"Training rows : {len(train)}")
    print(f"Testing rows  : {len(test)}")
    print(f"Accuracy      : {acc:.4f} ({acc*100:.2f}%)")

    # Confusion matrix
    slot_labels = encoders[TARGET_COL].classes_
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=slot_labels, columns=slot_labels).to_string())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=slot_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Random Forest – Confusion Matrix (time_slot)")
    plt.tight_layout()
    plt.savefig("outputs/rf_confusion_matrix.png", dpi=120)
    plt.close()
    print("Confusion matrix saved → outputs/rf_confusion_matrix.png")

    # Variable importance
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\nVariable Importance:")
    print(importances.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="bar", ax=ax)
    ax.set_title("Random Forest – Variable Importance (time_slot)")
    ax.set_ylabel("Mean Decrease in Impurity")
    ax.set_xlabel("Feature")
    plt.tight_layout()
    plt.savefig("outputs/rf_variable_importance.png", dpi=120)
    plt.close()
    print("Variable importance plot saved → outputs/rf_variable_importance.png")

    return clf, encoders, acc


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    train_random_forest()
