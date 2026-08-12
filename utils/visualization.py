"""
utils/visualization.py
------------------------
Plotting helpers used to generate figures for report.pdf.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """Plots and optionally saves a confusion matrix heatmap."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    return fig


def plot_training_curve(loss_history, title="Training Loss", save_path=None):
    """Plots hinge-loss vs epoch to show convergence (Part C/D requirement)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Hinge Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curve to {save_path}")
    return fig


def plot_confidence_distribution(correct_conf, incorrect_conf, save_path=None):
    """Histogram comparing confidence of correct vs incorrect predictions (Part E)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(correct_conf, bins=15, alpha=0.6, label="Correct", color="green")
    ax.hist(incorrect_conf, bins=15, alpha=0.6, label="Incorrect", color="red")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Number of predictions")
    ax.set_title("Confidence Distribution: Correct vs Incorrect")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confidence distribution to {save_path}")
    return fig
