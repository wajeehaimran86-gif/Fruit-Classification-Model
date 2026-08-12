"""
utils/helpers.py
------------------
Small generic utility functions shared across the project.
"""

import time
import numpy as np


class Timer:
    """Simple context manager to time a block of code.

    Usage:
        with Timer("Feature extraction"):
            ... do work ...
    """

    def __init__(self, label="Elapsed"):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"{self.label}: {elapsed:.2f}s")


def save_npz(filepath, **arrays):
    """Thin wrapper around np.savez with a print confirmation."""
    np.savez(filepath, **arrays)
    print(f"Saved: {filepath}")


def load_npz(filepath):
    """Thin wrapper around np.load with allow_pickle enabled."""
    return np.load(filepath, allow_pickle=True)


def train_test_counts_summary(class_names, counts_dict):
    """
    Pretty-prints a per-class, per-split summary table.
    counts_dict: {"train": {"apple": 233, ...}, "val": {...}, "test": {...}}
    """
    print(f"{'Class':12s} | {'Train':>6s} | {'Val':>6s} | {'Test':>6s} | {'Total':>6s}")
    print("-" * 50)
    for cls in class_names:
        train_c = counts_dict.get("train", {}).get(cls, 0)
        val_c = counts_dict.get("val", {}).get(cls, 0)
        test_c = counts_dict.get("test", {}).get(cls, 0)
        total = train_c + val_c + test_c
        print(f"{cls:12s} | {train_c:6d} | {val_c:6d} | {test_c:6d} | {total:6d}")
