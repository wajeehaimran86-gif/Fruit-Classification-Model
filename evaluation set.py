"""
evaluate_test_set.py
----------------------
Runs the trained linear SVM on the ENTIRE held-out test set and reports:
  - Overall test accuracy (needed for your report / rubric)
  - Accuracy broken down separately for Kaggle images vs your own
    phone photos (WhatsApp images) - to directly show the
    generalization gap discussed in your report.

Run from project root:
    python evaluate_test_set.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "features"))

from svm import OvRSVM
from confidence import predict_with_confidence, calibration_check
from feature_extractor import extract_features

DATA_DIR = os.path.join("data", "processed", "test")
CACHE_DIR = os.path.join("results", "saved_models")


def is_phone_photo(filename):
    """Your own photos were transferred via WhatsApp, so they contain
    'WhatsApp' in the filename - use that to separate the two sources."""
    return "whatsapp" in filename.lower()


def main():
    print("Loading trained linear SVM...")
    model = OvRSVM()
    model.load(os.path.join(CACHE_DIR, "linear_svm.npz"))

    X, y, sources = [], [], []  # sources: "kaggle" or "phone"

    class_names = sorted(os.listdir(DATA_DIR))
    for cls in class_names:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            try:
                feat = extract_features(fpath)
                X.append(feat)
                y.append(cls)
                sources.append("phone" if is_phone_photo(fname) else "kaggle")
            except Exception as e:
                print(f"  [skip] {fpath}: {e}")

    X = np.array(X)
    y = np.array(y)
    sources = np.array(sources)

    print(f"\nTotal test images: {len(y)} "
          f"(kaggle={np.sum(sources=='kaggle')}, phone={np.sum(sources=='phone')})\n")

    predictions, confidences = predict_with_confidence(model, X)
    predictions = np.array(predictions)

    overall_acc = np.mean(predictions == y)
    print(f"OVERALL TEST ACCURACY: {overall_acc:.4f}  ({overall_acc*100:.2f}%)")

    for src in ["kaggle", "phone"]:
        mask = sources == src
        if mask.sum() == 0:
            continue
        acc = np.mean(predictions[mask] == y[mask])
        print(f"  {src:8s} test accuracy: {acc:.4f}  ({acc*100:.2f}%)  [{mask.sum()} images]")

    print()
    calibration_check(y, predictions, confidences)

    # Per-class breakdown - useful for report's confusion analysis
    print("\nPer-class accuracy:")
    for cls in class_names:
        mask = y == cls
        if mask.sum() == 0:
            continue
        acc = np.mean(predictions[mask] == y[mask])
        print(f"  {cls:12s}: {acc:.4f}  ({mask.sum()} images)")


if __name__ == "__main__":
    main()