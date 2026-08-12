"""
shape_features.py
-------------------
Extracts shape-based features (circularity, aspect ratio) from an image
using background segmentation and contour detection.
"""

import cv2
import numpy as np


def extract_shape_features(img_bgr):
    """
    Segment the fruit from the background, find its contour, and
    compute shape descriptors: circularity and aspect ratio.

    Input: img_bgr - the original BGR image, shape (128,128,3)
    Output: 1D numpy array of length 2: [circularity, aspect_ratio]
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.array([0.0, 1.0])

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        circularity = 0.0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)

    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h != 0 else 1.0

    return np.array([circularity, aspect_ratio])
