"""
train_kernel.py
-----------------
Part D: Trains the kernelized (RBF) One-vs-Rest SVM using the CACHED
feature matrices saved earlier by train.py (avoids re-extracting
features from images, which is slow).

Run from project root, AFTER train.py has been run once:
    python train_kernel.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from kernel import KernelOvRSVM

CACHE_DIR = os.path.join("results", "saved_models")
MODEL_OUT = os.path.join(CACHE_DIR, "kernel_svm.npz")

# Hyperparameters - tune these and record results in your report
KERNEL_TYPE = "rbf"
C = 1.0
SIGMA = 5.0        # try 1, 5, 10 and compare - record in report
N_EPOCHS = 5       # kernel Pegasos epochs (kept small since it's slower than linear)


def main():
    print("Loading cached features (from train.py run)...")
    train_data = np.load(os.path.join(CACHE_DIR, "train_features.npz"))
    val_data = np.load(os.path.join(CACHE_DIR, "val_features.npz"))

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    print(f"Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples\n")

    # NOTE: Kernel SVM is O(n^2) in memory/compute for the kernel matrix.
    # With ~2900 train samples this is workable but slow. If it's too
    # slow on your machine, uncomment the lines below to subsample:
    #
    # subsample_size = 1000
    # idx = np.random.choice(len(X_train), subsample_size, replace=False)
    # X_train, y_train = X_train[idx], y_train[idx]
    # print(f"Subsampled training set to {subsample_size} for speed.")

    print(f"Training Kernel OvR SVM (kernel={KERNEL_TYPE}, C={C}, sigma={SIGMA})...")
    model = KernelOvRSVM(kernel=KERNEL_TYPE, C=C, n_epochs=N_EPOCHS, sigma=SIGMA)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    train_acc = np.mean(train_preds == y_train)
    print(f"\nKernel SVM Training accuracy: {train_acc:.4f}")

    val_preds = model.predict(X_val)
    val_acc = np.mean(val_preds == y_val)
    print(f"Kernel SVM Validation accuracy: {val_acc:.4f}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    model.save(MODEL_OUT)

    print("\n--- Compare this to your Linear SVM (Part C) results ---")
    print("Linear SVM was: Training=0.9722, Validation=0.9740")
    print("Write this comparison in your report.pdf (Part D requirement).")


if __name__ == "__main__":
    main()