# Softmax / Platt Scaling
"""
confidence.py
--------------
Part E: Converts raw OvR decision scores into a calibrated confidence
score (0-100%) using softmax over the K binary decision scores.

softmax(score_k) = exp(score_k) / sum(exp(score_j) for all j)
"""

import numpy as np


def softmax_confidence(decision_scores):
    """
    decision_scores: (n_samples, n_classes) raw OvR decision scores
    Returns: (n_samples, n_classes) probabilities that sum to 1 per row
    """
    # subtract max per row for numerical stability (avoids overflow in exp)
    shifted = decision_scores - np.max(decision_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def predict_with_confidence(model, X, class_names=None):
    """
    Runs the model on X and returns, for each sample:
      - predicted class name
      - confidence percentage (0-100) for that prediction

    Works with either OvRSVM (models/svm.py) or KernelOvRSVM (models/kernel.py)
    since both expose decision_scores() and classes_.
    """
    scores = model.decision_scores(X)
    probs = softmax_confidence(scores)

    best_idx = np.argmax(probs, axis=1)
    classes = class_names if class_names is not None else model.classes_

    predictions = [classes[i] for i in best_idx]
    confidences = [probs[row, i] * 100 for row, i in enumerate(best_idx)]

    return predictions, confidences


def calibration_check(y_true, predictions, confidences):
    """
    Part E requirement: check whether correct predictions tend to have
    HIGHER confidence than incorrect ones. Prints a simple summary table.
    """
    y_true = np.array(y_true)
    predictions = np.array(predictions)
    confidences = np.array(confidences)

    correct_mask = predictions == y_true
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    print("=" * 50)
    print("CONFIDENCE CALIBRATION CHECK")
    print("=" * 50)
    print(f"Number of correct predictions:   {len(correct_conf)}")
    print(f"Number of incorrect predictions: {len(incorrect_conf)}")

    if len(correct_conf) > 0:
        print(f"\nAvg confidence on CORRECT predictions:   {correct_conf.mean():.2f}%")
    if len(incorrect_conf) > 0:
        print(f"Avg confidence on INCORRECT predictions: {incorrect_conf.mean():.2f}%")

    if len(correct_conf) > 0 and len(incorrect_conf) > 0:
        if correct_conf.mean() > incorrect_conf.mean():
            print("\n[GOOD] Model is well-calibrated: correct predictions have higher confidence.")
        else:
            print("\n[WARNING] Model is poorly calibrated: incorrect predictions have "
                  "similar/higher confidence than correct ones.")

    return {
        "correct_conf_mean": correct_conf.mean() if len(correct_conf) else None,
        "incorrect_conf_mean": incorrect_conf.mean() if len(incorrect_conf) else None,
    }