"""
train.py
--------
Original Part-C training entry point, upgraded only at the data pipeline:
foreground isolation + object crop + five synthetic backgrounds + camera
augmentation + feature scaling + class balancing.  The custom One-vs-Rest
linear SVM is unchanged.

Run:
    python train.py
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "models"))
sys.path.append(os.path.join(ROOT, "features"))
sys.path.append(ROOT)

from robust_pipeline import training_features
from svm import OvRSVM

DATA_DIR = os.path.join(ROOT, "data", "processed")
MODEL_OUT = os.path.join(ROOT, "results", "saved_models", "linear_svm.npz")
SCALER_OUT = os.path.join(ROOT, "results", "saved_models", "feature_scaler.npz")
TRAIN_CACHE = os.path.join(ROOT, "results", "saved_models", "train_features.npz")
VAL_CACHE = os.path.join(ROOT, "results", "saved_models", "val_features.npz")

# Sir's requested focused model: 3 fruits.
CLASS_NAMES = ["apple", "banana", "orange"]
VIEWS_PER_IMAGE = 6
MAX_BASE_IMAGES_PER_CLASS = 60
MAX_VAL_IMAGES_PER_CLASS = 15
LEARNING_RATE = 0.0005
C = 1.0
N_EPOCHS = 100
SEED = 42


def _image_files(folder):
    return sorted([f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])


def _select_files(files, max_n, seed):
    """Keep user photos, then use a deterministic subset for reproducibility."""
    files = sorted(files)
    if len(files) <= max_n:
        return files
    real = [f for f in files if any(t in f.lower() for t in ("whatsapp", "img_", "photo"))]
    real = real[:max_n]
    remaining = [f for f in files if f not in set(real)]
    need = max_n - len(real)
    # Deterministic first-N controlled images avoids expensive/corrupt outliers
    # while preserving the original dataset ordering.
    return real + remaining[:need]


def load_split(split_name, augment=False):
    split_dir = os.path.join(DATA_DIR, split_name)
    X, y = [], []
    for ci, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        files = _image_files(cls_dir)
        if augment:
            files = _select_files(files, MAX_BASE_IMAGES_PER_CLASS, SEED + ci * 101)
        else:
            files = _select_files(files, MAX_VAL_IMAGES_PER_CLASS, SEED + 5000 + ci * 101)
        print(f"  [{split_name}] {cls}: {len(files)} base images"
              + (f" -> {len(files) * VIEWS_PER_IMAGE} augmented views" if augment else ""))

        for j, fname in enumerate(files):
            path = os.path.join(cls_dir, fname)
            try:
                if augment:
                    feats = training_features(path, seed_base=SEED + j * 17 + ci * 10000)
                    X.append(feats)
                    y.extend([cls] * len(feats))
                else:
                    # Validation remains one deterministic, background-masked view.
                    from features.feature_extractor import extract_features
                    X.append(extract_features(path))
                    y.append(cls)
            except Exception as e:
                print(f"    [skip] {path}: {e}")

    if augment:
        return np.vstack(X).astype(np.float32), np.array(y)
    return np.asarray(X, dtype=np.float32), np.array(y)


def fit_scaler(X):
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_scaler(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def balance_classes(X, y):
    rng = np.random.default_rng(SEED)
    counts = {c: int(np.sum(y == c)) for c in CLASS_NAMES}
    target = min(counts.values())
    idx = []
    for c in CLASS_NAMES:
        ci = np.where(y == c)[0]
        if len(ci) > target:
            ci = rng.choice(ci, target, replace=False)
        idx.extend(ci.tolist())
    idx = np.asarray(idx)
    rng.shuffle(idx)
    print("Balanced training views:", {c: int(np.sum(y[idx] == c)) for c in CLASS_NAMES})
    return X[idx], y[idx]


def main():
    print("=" * 70)
    print("BACKGROUND-INVARIANT FRUIT CLASSIFICATION TRAINING")
    print("Classes:", CLASS_NAMES)
    print("=" * 70)

    print("\n1) Building augmented TRAIN features...")
    X_train, y_train = load_split("train", augment=True)
    print("Augmented train matrix:", X_train.shape)

    X_train, y_train = balance_classes(X_train, y_train)

    print("\n2) Building VAL features...")
    X_val, y_val = load_split("val", augment=False)
    print("Validation matrix:", X_val.shape)

    print("\n3) Standardizing features...")
    mean, std = fit_scaler(X_train)
    X_train_s = apply_scaler(X_train, mean, std)
    X_val_s = apply_scaler(X_val, mean, std)

    print("\n4) Training original custom One-vs-Rest Linear SVM...")
    model = OvRSVM(learning_rate=LEARNING_RATE, C=C, n_epochs=N_EPOCHS)
    model.fit(X_train_s, y_train)

    train_preds = model.predict(X_train_s)
    val_preds = model.predict(X_val_s)
    print(f"\nTraining accuracy:   {np.mean(train_preds == y_train):.4f}")
    print(f"Validation accuracy: {np.mean(val_preds == y_val):.4f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    np.savez(SCALER_OUT, mean=mean, std=std)
    np.savez(TRAIN_CACHE, X=X_train_s, y=y_train)
    np.savez(VAL_CACHE, X=X_val_s, y=y_val)
    print("Saved model:", MODEL_OUT)
    print("Saved scaler:", SCALER_OUT)


if __name__ == "__main__":
    main()
