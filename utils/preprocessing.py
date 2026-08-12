"""
utils/preprocessing.py
------------------------
Helper functions for splitting a dataset into train/val/test,
used by preprocess.py.
"""

import random


def split_filenames(files, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Shuffles a list of filenames and splits it into three lists:
    (train_files, val_files, test_files), according to the given ratios.
    The remainder after train+val goes to test.
    """
    files = files[:]
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def check_class_balance(class_counts, min_required=80):
    """
    Given a dict of class_name -> image_count, returns a list of
    warning messages for any class below the minimum required count.
    """
    warnings = []
    for cls, count in class_counts.items():
        if count < min_required:
            warnings.append(
                f"Class '{cls}' has only {count} images "
                f"(minimum required: {min_required})"
            )
    return warnings
