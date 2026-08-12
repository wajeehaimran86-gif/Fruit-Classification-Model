"""
color_features.py
-------------------
Extracts color-based features from an HSV image using histograms.
"""

import cv2
import numpy as np


def extract_color_features(img_hsv, bins=16):
    """
    Extract a normalized histogram for each of the H, S, V channels
    and concatenate them into one color feature vector.

    Input: img_hsv - image already converted to HSV, shape (128,128,3)
    Output: 1D numpy array of length bins*3 (e.g. 48 numbers if bins=16)
    """
    hist_h = cv2.calcHist([img_hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([img_hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([img_hsv], [2], None, [bins], [0, 256])

    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
    hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)

    color_features = np.concatenate([hist_h, hist_s, hist_v])
    return color_features
