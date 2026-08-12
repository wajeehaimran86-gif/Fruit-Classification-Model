"""
utils/image_utils.py
----------------------
Small helper functions for image I/O and validation, used by the
data pipeline (preprocess.py, clean_data.py).
"""

import os
import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def is_image_file(filename):
    """True if the filename has a supported image extension."""
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def load_image(path, target_size=None):
    """
    Loads an image with OpenCV, optionally resizing it.
    Returns None (instead of raising) if the file can't be read,
    so callers can skip corrupt files gracefully.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    if target_size is not None:
        img = cv2.resize(img, target_size)
    return img


def list_images_in_dir(directory):
    """Returns a list of full paths to all image files in a directory."""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if is_image_file(f)
    ]


def image_brightness(img_bgr):
    """Average pixel brightness (0-255), useful for detecting very
    dark/bright (likely broken) images during cleaning."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))
