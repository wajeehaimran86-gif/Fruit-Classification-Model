"""
utils/metrics.py
-----------------
Simple evaluation metric helpers (no sklearn classifiers used -
these are just arithmetic on predictions, which is allowed).
"""

import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of predictions that match the true label."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, class_names):
    """
    Builds an (n_classes, n_classes) confusion matrix.
    Rows = true label, Columns = predicted label.
    """
    idx = {cls: i for i, cls in enumerate(class_names)}
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def per_class_accuracy(y_true, y_pred, class_names):
    """Returns a dict: class_name -> accuracy for that class only."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    result = {}
    for cls in class_names:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        result[cls] = np.mean(y_pred[mask] == y_true[mask])
    return result


def precision_recall_f1(y_true, y_pred, class_names):
    """
    Computes per-class precision, recall, and F1 score (one-vs-rest style).
    Returns a dict: class_name -> {precision, recall, f1}
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    results = {}

    for cls in class_names:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        results[cls] = {"precision": precision, "recall": recall, "f1": f1}

    return results
