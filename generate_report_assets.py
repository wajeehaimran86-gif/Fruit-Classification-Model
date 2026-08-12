"""
generate_report_assets.py
----------------------------
Quick script to generate the remaining report assets:
  1. results/predictions.csv - every test image, true label, predicted
     label, and confidence
  2. results/confusion_matrix/confusion_matrix.png - visual confusion
     matrix for the report

Run from project root:
    python generate_report_assets.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "features"))

from svm import OvRSVM
from confidence import predict_with_confidence
from feature_extractor import extract_features

DATA_DIR = os.path.join("data", "processed", "test")
CACHE_DIR = os.path.join("results", "saved_models")
PRED_OUT = os.path.join("results", "predictions.csv")
CM_OUT = os.path.join("results", "confusion_matrix", "confusion_matrix.png")


def main():
    print("Loading trained linear SVM...")
    model = OvRSVM()
    model.load(os.path.join(CACHE_DIR, "linear_svm.npz"))

    class_names = sorted(os.listdir(DATA_DIR))

    rows = []
    X_list, y_list = [], []

    print("Extracting features for all test images...")
    for cls in class_names:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            try:
                feat = extract_features(fpath)
                X_list.append(feat)
                y_list.append(cls)
                rows.append({"image_path": fpath, "true_label": cls})
            except Exception as e:
                print(f"  [skip] {fpath}: {e}")

    X = np.array(X_list)
    y_true = np.array(y_list)

    print("Predicting...")
    predictions, confidences = predict_with_confidence(model, X)

    for i, row in enumerate(rows):
        row["predicted_label"] = predictions[i]
        row["confidence_percent"] = round(confidences[i], 2)
        row["correct"] = predictions[i] == row["true_label"]

    # --- Save predictions.csv ---
    os.makedirs(os.path.dirname(PRED_OUT), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(PRED_OUT, index=False)
    print(f"Saved: {PRED_OUT} ({len(df)} rows)")

    overall_acc = df["correct"].mean()
    print(f"Overall test accuracy: {overall_acc:.4f}")

    # --- Build confusion matrix manually (no sklearn classifier used) ---
    n = len(class_names)
    idx = {cls: i for i, cls in enumerate(class_names)}
    cm = np.zeros((n, n), dtype=int)
    for true_label, pred_label in zip(y_true, predictions):
        cm[idx[true_label], idx[pred_label]] += 1

    # --- Plot confusion matrix ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Linear SVM (Test Set)")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    os.makedirs(os.path.dirname(CM_OUT), exist_ok=True)
    plt.savefig(CM_OUT, dpi=150)
    print(f"Saved: {CM_OUT}")


if __name__ == "__main__":
    main()
