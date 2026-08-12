"""
texture_features.py
---------------------
Extracts texture-based features using edge density (Canny edge detector).
"""

import cv2
import numpy as np


def extract_texture_features(img_bgr):
    """
    Measure texture roughness using edge density.

    Input: img_bgr - the original BGR image
    Output: 1D numpy array of length 1: [edge_density]
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)
    return np.array([edge_density])
