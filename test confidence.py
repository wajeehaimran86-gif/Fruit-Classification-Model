"""
test_confidence.py
--------------------
Quick script to test Part E: loads the saved linear SVM model +
cached validation features, computes confidence scores, and runs
the calibration check.

Run from project root:
    python test_confidence.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from svm import OvRSVM
from confidence import predict_with_confidence, calibration_check

CACHE_DIR = os.path.join("results", "saved_models")


def main():
    print("Loading saved linear SVM model...")
    model = OvRSVM()
    model.load(os.path.join(CACHE_DIR, "linear_svm.npz"))

    print("Loading validation features...")
    val_data = np.load(os.path.join(CACHE_DIR, "val_features.npz"))
    X_val, y_val = val_data["X"], val_data["y"]

    print("Computing predictions with confidence...")
    predictions, confidences = predict_with_confidence(model, X_val)

    # Show a few example predictions
    print("\nSample predictions:")
    for i in range(10):
        mark = "✓" if predictions[i] == y_val[i] else "✗"
        print(f"  {mark} true={y_val[i]:12s} pred={predictions[i]:12s} confidence={confidences[i]:.1f}%")

    # Full calibration check (Part E requirement)
    calibration_check(y_val, predictions, confidences)


if __name__ == "__main__":
    main()