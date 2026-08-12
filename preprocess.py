"""
preprocess.py
--------------
Part A helper: resizes raw images and splits them into
Training (70%) / Validation (15%) / Test (15%) folders.

Usage:
    Put your raw images like this first:

        data/raw/apple/*.jpg
        data/raw/banana/*.jpg
        data/raw/orange/*.jpg
        data/raw/mango/*.jpg
        data/raw/grapes/*.jpg
        data/raw/strawberry/*.jpg

    Then run:
        python preprocess.py

    Output goes to:
        data/processed/train/<class>/
        data/processed/val/<class>/
        data/processed/test/<class>/
"""

import os
import random
import shutil
from PIL import Image

# ---- CONFIG ----
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
IMG_SIZE = (128, 128)          # resize target
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)


def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def resize_and_save(src_path, dst_path):
    """Open an image, convert to RGB, resize, and save."""
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(dst_path)
        return True
    except Exception as e:
        print(f"  [skip] Could not process {src_path}: {e}")
        return False


def split_list(files, train_ratio, val_ratio):
    """Shuffle and split a list of filenames into train/val/test."""
    files = files[:]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"Raw data folder not found: {RAW_DIR}")
        print("Create data/raw/<class_name>/ folders and put images inside first.")
        return

    class_names = sorted(
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    )

    if not class_names:
        print(f"No class folders found inside {RAW_DIR}")
        return

    print(f"Found {len(class_names)} classes: {class_names}\n")

    # make output dirs
    for split in ["train", "val", "test"]:
        for cls in class_names:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

    summary = {}

    for cls in class_names:
        cls_raw_dir = os.path.join(RAW_DIR, cls)
        files = [f for f in os.listdir(cls_raw_dir) if is_image_file(f)]

        if len(files) < 80:
            print(f"⚠ Warning: class '{cls}' has only {len(files)} images "
                  f"(assignment requires at least 80).")

        train_files, val_files, test_files = split_list(
            files, TRAIN_RATIO, VAL_RATIO
        )

        counts = {"train": 0, "val": 0, "test": 0}

        for split_name, split_files in [
            ("train", train_files), ("val", val_files), ("test", test_files)
        ]:
            for fname in split_files:
                src = os.path.join(cls_raw_dir, fname)
                dst = os.path.join(OUT_DIR, split_name, cls, fname)
                if resize_and_save(src, dst):
                    counts[split_name] += 1

        summary[cls] = counts
        print(f"{cls:15s} -> train={counts['train']:3d}  "
              f"val={counts['val']:3d}  test={counts['test']:3d}")

    print("\nDone. Processed images are in:", OUT_DIR)
    print("\nSummary:")
    for cls, counts in summary.items():
        total = sum(counts.values())
        print(f"  {cls}: {total} images total")


if __name__ == "__main__":
    main()
