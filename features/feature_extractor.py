"""
feature_extractor.py
--------------------
Master feature extractor.  The original 51-dimensional feature design is
retained, but features are now computed from a foreground-isolated fruit so
that table/wall/window backgrounds do not become class signals.
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from robust_pipeline import foreground_mask, object_crop, extract_masked_features


def extract_features(image_path, resize_to=(128, 128), bins=16):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    mask = foreground_mask(img)
    obj, om = object_crop(img, mask, out_size=resize_to)
    return extract_masked_features(obj, om, resize_to=resize_to, bins=bins)


if __name__ == "__main__":
    test_path = "../data/processed/train/apple/Apple_Braeburn_1_r1_46.jpg"
    vec = extract_features(test_path)
    print("Feature vector length:", len(vec))
    print("Feature vector:", vec)
