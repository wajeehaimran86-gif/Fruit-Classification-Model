# Hyperparameters
"""
config.py
----------
Central configuration for hyperparameters used across the project.
Import from here instead of hardcoding values in multiple files.
"""

# Data
IMAGE_SIZE = (128, 128)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature extraction (Part B)
COLOR_HIST_BINS = 16

# Linear SVM (Part C)
LINEAR_LEARNING_RATE = 0.001
LINEAR_C = 1.0
LINEAR_EPOCHS = 300

# Kernel SVM (Part D)
KERNEL_TYPE = "rbf"          # 'rbf' or 'poly'
KERNEL_C = 1.0
KERNEL_SIGMA = 5.0           # best value found during tuning (5.0 > 15.0)
KERNEL_EPOCHS = 5
POLY_DEGREE = 3
POLY_C = 1.0

# Class names (must match folder names in data/raw and data/processed)
CLASS_NAMES = ["apple", "banana", "orange"]
