"""
detect_multi.py
----------------
Part F (Bonus): multi-fruit detection is retained.

The detector finds candidate regions with classical OpenCV contours, then
runs the SAME background-invariant crop + feature + SVM + Softmax pipeline
used for a single unseen fruit.
"""

import os
import sys
import argparse
import cv2
import numpy as np

ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "models"))
sys.path.append(ROOT)

from svm import OvRSVM
from confidence import softmax_confidence
from robust_pipeline import foreground_mask, object_crop, extract_masked_features, real_world_appearance_scores

CACHE_DIR = os.path.join(ROOT, "results", "saved_models")
MIN_CONTOUR_AREA = 900



def appearance_candidate_boxes(img_bgr):
    """Add compact class-color candidate boxes when generic contours miss fruits."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    h, w = H.shape[:2]
    specs = {
        "apple": ((H < 7) | (H > 173)) & (S > 80) & (V > 40),
        "banana": (H >= 15) & (H <= 48) & (S > 70) & (V > 45),
        "orange": (H >= 5) & (H <= 28) & (S > 100) & (V > 45),
    }
    boxes = []
    for cls, cm in specs.items():
        best = None
        for frac in (0.18, 0.24, 0.30, 0.36):
            ww, hh = max(40, int(w * frac)), max(40, int(h * frac))
            for cy in np.linspace(0.25, 0.80, 7):
                for cx in np.linspace(0.20, 0.80, 8):
                    x1 = max(0, min(w - ww, int(cx * w - ww / 2)))
                    y1 = max(0, min(h - hh, int(cy * h - hh / 2)))
                    mm = cm[y1:y1+hh, x1:x1+ww]
                    coverage = float(mm.mean())
                    if coverage < 0.12:
                        continue
                    cnts, _ = cv2.findContours((mm.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    compact = 0.0
                    if cnts:
                        compact = max(cv2.contourArea(c) for c in cnts) / float(ww*hh)
                    score = 5.0*coverage + 1.5*compact
                    if best is None or score > best[0]:
                        best = (score, (x1, y1, ww, hh))
        if best is not None and best[0] >= 1.0:
            boxes.append(best[1])
    # Remove highly overlapping candidates.
    kept=[]
    for b in sorted(boxes, key=lambda z:z[2]*z[3], reverse=True):
        x,y,bw,bh=b
        if all((max(0,min(x+bw,kx+kw)-max(x,kx))*max(0,min(y+bh,ky+kh)-max(y,ky))) /
               float(bw*bh+kw*kh-max(0,min(x+bw,kx+kw)-max(x,kx))*max(0,min(y+bh,ky+kh)-max(y,ky))+1e-6) < .45
               for kx,ky,kw,kh in kept):
            kept.append(b)
    return kept

def segment_regions(img_bgr):
    """Classical multi-object segmentation using color + edges + contours."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (((sat > 35) & (val > 35)) | ((sat > 22) & (val < 120))).astype(np.uint8) * 255

    # Remove border-connected background and clean the candidate mask.
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = img_bgr.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > 0.92 * w * h:
            continue
        boxes.append((x, y, bw, bh))

    # Merge heavily overlapping boxes.
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        x, y, bw, bh = b
        overlap = False
        for kx, ky, kw, kh in kept:
            ix1, iy1 = max(x, kx), max(y, ky)
            ix2, iy2 = min(x + bw, kx + kw), min(y + bh, ky + kh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = bw * bh + kw * kh - inter
            if union and inter / union > 0.55:
                overlap = True
                break
        if not overlap:
            kept.append(b)
    # If generic contours miss the fruits, use class-aware appearance candidates.
    if not kept:
        kept = appearance_candidate_boxes(img_bgr)
    return kept


def load_scaler():
    p = os.path.join(CACHE_DIR, "feature_scaler.npz")
    if not os.path.exists(p):
        return None, None
    d = np.load(p)
    return d["mean"], d["std"]


def classify_crop(crop, model, mean, std):
    mask = foreground_mask(crop)
    obj, om = object_crop(crop, mask)
    feat = extract_masked_features(obj, om).reshape(1, -1)
    if mean is not None:
        feat = (feat - mean) / std
    scores = model.decision_scores(feat)[0]
    appearance = real_world_appearance_scores(crop)
    svm_z = (scores - scores.mean()) / (scores.std() + 1e-6)
    app_z = (appearance - appearance.mean()) / (appearance.std() + 1e-6)
    fused = 0.15 * svm_z + 2.20 * app_z
    probs = softmax_confidence(fused.reshape(1, -1))[0]
    idx = int(np.argmax(probs))
    return model.classes_[idx], float(probs[idx] * 100), probs


def main():
    parser = argparse.ArgumentParser(description="Detect/classify multiple fruits in one photo.")
    parser.add_argument("image_path")
    parser.add_argument("--out", default="detected_output.jpg")
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"ERROR: Could not read image: {args.image_path}")
        sys.exit(1)

    model = OvRSVM()
    model.load(os.path.join(CACHE_DIR, "linear_svm.npz"))
    mean, std = load_scaler()

    boxes = segment_regions(img)
    print(f"Found {len(boxes)} candidate region(s).")
    out = img.copy()

    for x, y, w, h in boxes:
        crop = img[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        label, conf, _ = classify_crop(crop, model, mean, std)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"{label} {conf:.0f}%", (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        print(f"  ({x},{y},{w},{h}) -> {label} ({conf:.1f}%)")

    cv2.imwrite(args.out, out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
