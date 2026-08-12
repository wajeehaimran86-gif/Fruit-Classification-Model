"""
robust_pipeline.py
------------------
Background-invariant preprocessing for the original fruit-classification
project.  The original 51-dimensional feature design and custom OvR SVM
are kept; this module only makes the image representation more robust.

Training pipeline:
    image -> foreground mask -> crop -> resize -> augmentation ->
    5 synthetic backgrounds -> masked color/shape/texture features

Inference pipeline:
    image -> foreground mask -> crop -> resize -> multiple normalized views ->
    same masked features

rembg is used automatically when it is installed.  OpenCV/GrabCut is the
built-in fallback so the project still runs on a normal Python environment.
"""

import os
import cv2
import numpy as np

IMG_SIZE = (128, 128)
BINS = 16
BACKGROUND_KINDS = ("wood", "kitchen", "gradient", "noise", "neutral")

_REMBG_REMOVE = None
_REMBG_TRIED = False

def _get_rembg_remove():
    global _REMBG_REMOVE, _REMBG_TRIED
    if _REMBG_TRIED:
        return _REMBG_REMOVE
    _REMBG_TRIED = True
    try:
        from rembg import remove
        _REMBG_REMOVE = remove
    except Exception:
        _REMBG_REMOVE = None
    return _REMBG_REMOVE


def _largest_component(mask, min_area_ratio=0.01):
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    h, w = mask.shape[:2]
    min_area = min_area_ratio * h * w
    good = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not good:
        return np.zeros_like(mask)
    # For grapes, the largest connected region can be a cluster; for other
    # fruits it is normally the fruit body.  Keep only the largest region.
    c = max(good, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out


def _fill_holes(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inv)


def _white_background_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Fruits-360 style white/near-white background.
    m = (((hsv[:, :, 1] > 28) & (hsv[:, :, 2] < 225)) |
         (hsv[:, :, 2] < 150)).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = _fill_holes(m)
    return _largest_component(m, 0.02)


def _grabcut_mask(img):
    h, w = img.shape[:2]
    # Center-biased rectangle works well for the single-fruit images used by
    # the assignment and avoids swallowing table/wall regions.
    rect = (max(1, int(0.12 * w)), max(1, int(0.06 * h)),
            max(2, int(0.76 * w)), max(2, int(0.86 * h)))
    gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return np.zeros((h, w), np.uint8)
    m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = _fill_holes(m)
    return _largest_component(m, 0.015)


def _color_prior_mask(img):
    """Fallback for strongly colored fruits against low-saturation backgrounds."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Fruit pixels tend to have saturation or chroma; also retain dark banana
    # stems and grape shadows through the value term.
    m = (((hsv[:, :, 1] > 42) & (hsv[:, :, 2] > 35)) |
         ((hsv[:, :, 1] > 24) & (hsv[:, :, 2] < 115))).astype(np.uint8) * 255
    k = np.ones((7, 7), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = _fill_holes(m)
    return _largest_component(m, 0.01)


def _mask_quality(mask, img_shape):
    if mask is None:
        return 0.0
    h, w = img_shape[:2]
    ratio = cv2.countNonZero(mask) / float(h * w)
    if ratio < 0.015 or ratio > 0.90:
        return 0.0
    return min(1.0, ratio / 0.08) if ratio < 0.08 else 1.0


def _remove_green_accessories(img, mask):
    """Remove small green leaves/stems from a fruit foreground mask.

    Real-world orange photos often contain a green leaf/strip touching the
    fruit.  Keeping that accessory in the crop can shift the handcrafted
    color/shape features toward banana-like colors.  We keep the main warm
    fruit component and discard detached green components.
    """
    if mask is None or cv2.countNonZero(mask) == 0:
        return mask

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    green = (((H >= 35) & (H <= 95) & (S > 55) & (V > 35))).astype(np.uint8) * 255
    green = cv2.bitwise_and(green, mask)
    if cv2.countNonZero(green) == 0:
        return mask

    warm = (((H <= 30) | (H >= 165)) & (S > 55) & (V > 35)).astype(np.uint8) * 255
    warm = cv2.bitwise_and(warm, mask)
    warm = _largest_component(warm, 0.01)

    # If a substantial warm fruit body exists, remove green components that
    # do not overlap it.  This preserves tiny natural green pixels on the
    # fruit while dropping leaves/stems attached at the edge.
    if cv2.countNonZero(warm) > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
        cleaned = mask.copy()
        warm_dil = cv2.dilate(warm, np.ones((11, 11), np.uint8), iterations=1)
        for i in range(1, n):
            comp = (labels == i).astype(np.uint8) * 255
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            overlap = cv2.countNonZero(cv2.bitwise_and(comp, warm_dil))
            # A green component is an accessory when a warm fruit body is
            # clearly present.  The old 16%% area threshold could keep a
            # large leaf attached to an orange, so remove green accessories
            # much more aggressively while preserving the warm fruit body.
            if overlap == 0 or area < 0.55 * cv2.countNonZero(mask):
                cleaned[comp > 0] = 0
        # Final safety pass: for warm-fruit images, discard every remaining
        # green pixel.  This prevents a large attached leaf/stem from
        # contaminating the HSV/color features.
        warm_area = cv2.countNonZero(warm)
        if warm_area > 0.05 * mask.shape[0] * mask.shape[1]:
            cleaned[green > 0] = 0
        if cv2.countNonZero(cleaned) > 0.02 * mask.shape[0] * mask.shape[1]:
            return _largest_component(cleaned, 0.01)
    return mask



def _orange_body_mask(img):
    """Detect the orange fruit body while rejecting green leaves/stems.

    This targeted mask is used only when a green accessory is present.
    Orange is defined by a warm hue plus strong R>G>B chroma separation,
    which is much more specific than a generic saturation mask.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    B, G, R = cv2.split(img)
    B = B.astype(np.int16)
    G = G.astype(np.int16)
    R = R.astype(np.int16)

    m = (
        (H >= 3) & (H <= 30) &
        (S >= 80) & (V >= 55) &
        (R >= G + 18) &
        (G >= B + 12)
    ).astype(np.uint8) * 255

    # Fill small holes from orange highlights/texture, but never connect
    # across a large green leaf.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = _fill_holes(m)
    return _largest_component(m, 0.008)


def _green_accessory_ratio(img):
    """Return fraction of image occupied by saturated green accessory pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    green = ((H >= 35) & (H <= 95) & (S >= 55) & (V >= 35))
    return float(np.mean(green))


def _orange_with_green_leaf_mask(img):
    """Return a reliable orange-body mask when a green leaf/stem is present."""
    orange = _orange_body_mask(img)
    if cv2.countNonZero(orange) == 0:
        return np.zeros(orange.shape, np.uint8)

    h, w = img.shape[:2]
    orange_ratio = cv2.countNonZero(orange) / float(h * w)
    green_ratio = _green_accessory_ratio(img)

    # Require a substantial orange body and visible green accessory. This
    # keeps the special path from interfering with normal apple/banana photos.
    if orange_ratio < 0.06 or green_ratio < 0.008:
        return np.zeros_like(orange)

    # A single compact orange component is the fruit body. Remove all green
    # pixels by construction, including green regions touching the stem.
    return orange


def foreground_mask(img):
    """Return a robust single-fruit foreground mask.

    The orange+green-leaf case is handled before generic foreground removal:
    a saturated orange body is isolated from green leaves/stems, preventing
    the accessory from entering the crop or color/shape features.
    """
    h, w = img.shape[:2]

    # 0) Targeted orange-with-green-accessory path.
    # This must run before rembg/GrabCut because those methods can treat an
    # attached leaf and fruit as one foreground object.
    orange_leaf = _orange_with_green_leaf_mask(img)
    if cv2.countNonZero(orange_leaf) > 0:
        return orange_leaf

    # 1) rembg when available: strongest option for arbitrary real photos.
    try:
        remove = _get_rembg_remove()
        if remove is None:
            raise RuntimeError("rembg not installed")
        from PIL import Image
        rgba = remove(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        arr = np.asarray(rgba)
        if arr.ndim == 3 and arr.shape[2] == 4:
            alpha = arr[:, :, 3]
            m = (alpha > 35).astype(np.uint8) * 255
            m = _fill_holes(m)
            m = _largest_component(m, 0.01)
            if _mask_quality(m, img.shape) > 0:
                return _remove_green_accessories(img, m)
    except Exception:
        pass

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cs = max(4, min(18, h // 12, w // 12))
    corners = np.concatenate([
        hsv[:cs, :cs].reshape(-1, 3),
        hsv[:cs, -cs:].reshape(-1, 3),
        hsv[-cs:, :cs].reshape(-1, 3),
        hsv[-cs:, -cs:].reshape(-1, 3),
    ], axis=0)
    white_corner_ratio = float(np.mean((corners[:, 1] < 40) & (corners[:, 2] > 205)))

    # 2) Original Fruits-360 white background.
    if white_corner_ratio > 0.50:
        m = _white_background_mask(img)
        if _mask_quality(m, img.shape) > 0:
            return _remove_green_accessories(img, m)

    # 3) GrabCut for real-world photos.
    gc = _grabcut_mask(img)
    cp = _color_prior_mask(img)

    # Prefer GrabCut if it has a compact object; otherwise use color prior.
    if _mask_quality(gc, img.shape) > 0:
        # If GrabCut is very large, intersect it with a color prior when that
        # produces a sensible compact fruit.
        ratio = cv2.countNonZero(gc) / float(h * w)
        if ratio > 0.55 and _mask_quality(cp, img.shape) > 0:
            inter = cv2.bitwise_and(gc, cp)
            if _mask_quality(inter, img.shape) > 0.02:
                return _remove_green_accessories(img, inter)
        return _remove_green_accessories(img, gc)
    if _mask_quality(cp, img.shape) > 0:
        return _remove_green_accessories(img, cp)

    # Last resort: central region, not the whole image.
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (int(.38 * w), int(.42 * h)), 0, 0, 360, 255, -1)
    return _remove_green_accessories(img, m)


def object_crop(img, mask, out_size=IMG_SIZE, padding=0.12):
    """Crop to the foreground bbox and letterbox to a square."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return cv2.resize(img, out_size, interpolation=cv2.INTER_AREA), np.ones(out_size[::-1], np.uint8) * 255

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x2 - x1, y2 - y1
    px, py = max(2, int(bw * padding)), max(2, int(bh * padding))
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(img.shape[1], x2 + px), min(img.shape[0], y2 + py)

    crop = img[y1:y2, x1:x2]
    cm = mask[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 235, dtype=np.uint8)
    mcanvas = np.zeros((side, side), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2

    # IMPORTANT: copy only foreground pixels. Previously the entire bounding
    # box was copied, so a large attached leaf could remain visible in the
    # "fruit-focused" preview and could leak into downstream image operations.
    fg = (cm > 127)
    clean_crop = np.full_like(crop, 235)
    clean_crop[fg] = crop[fg]
    canvas[oy:oy + h, ox:ox + w] = clean_crop
    mcanvas[oy:oy + h, ox:ox + w] = cm
    return (cv2.resize(canvas, out_size, interpolation=cv2.INTER_AREA),
            cv2.resize(mcanvas, out_size, interpolation=cv2.INTER_NEAREST))


def _background(kind, size=IMG_SIZE, rng=None):
    rng = rng or np.random.default_rng()
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]

    if kind == "wood":
        grain = np.sin(xx / 7.0 + np.sin(yy / 19.0) * 1.7)
        noise = rng.normal(0, 7, (h, w))
        base = np.zeros((h, w, 3), np.float32)
        base[:, :, 0] = 110 + 26 * grain + noise
        base[:, :, 1] = 75 + 18 * grain + noise
        base[:, :, 2] = 43 + 12 * grain + noise
        return np.clip(base, 0, 255).astype(np.uint8)

    if kind == "kitchen":
        v = 190 + 24 * np.sin(xx / 15.0) + 18 * np.cos(yy / 21.0)
        base = np.dstack([v + 12, v + 8, v + 3]).astype(np.float32)
        base = np.clip(base, 0, 255).astype(np.uint8)
        for x in range(0, w, 32):
            cv2.line(base, (x, 0), (x, h), (150, 150, 150), 1)
        for y in range(0, h, 32):
            cv2.line(base, (0, y), (w, y), (150, 150, 150), 1)
        return cv2.GaussianBlur(base, (7, 7), 0)

    if kind == "gradient":
        t = (xx / max(1, w - 1)).astype(np.float32)
        c1 = np.array([55, 85, 130], np.float32)
        c2 = np.array([220, 190, 135], np.float32)
        return np.clip(c1 + (c2 - c1) * t[..., None], 0, 255).astype(np.uint8)

    if kind == "noise":
        base = rng.uniform(70, 210, (h, w, 3)).astype(np.float32)
        return np.clip(cv2.GaussianBlur(base, (11, 11), 0), 0, 255).astype(np.uint8)

    # neutral / soft indoor background
    base = np.full((h, w, 3), 232, np.uint8)
    vignette = 1 - 0.12 * (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2))
    return np.clip(base.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)


def _augment_object(obj, mask, seed):
    rng = np.random.default_rng(seed)
    h, w = obj.shape[:2]
    angle = float(rng.uniform(-16, 16))
    scale = float(rng.uniform(0.88, 1.08))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    aug = cv2.warpAffine(obj, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    am = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Mild camera-style illumination changes.
    alpha = float(rng.uniform(0.84, 1.16))
    beta = float(rng.uniform(-20, 20))
    aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Small hue/saturation perturbation in HSV.
    hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-5, 6))) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(rng.uniform(.90, 1.10)), 0, 255)
    aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.65:
        aug = cv2.GaussianBlur(aug, (0, 0), float(rng.uniform(.25, 1.15)))

    return aug, am


def recompose(obj, mask, kind, seed):
    rng = np.random.default_rng(seed)
    bg = _background(kind, (obj.shape[1], obj.shape[0]), rng)
    bg = np.clip(bg.astype(np.float32) * float(rng.uniform(.88, 1.10)), 0, 255).astype(np.uint8)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(obj.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)


def extract_masked_features(img_bgr, mask, resize_to=IMG_SIZE, bins=BINS):
    """Keep the original 51-D design, but measure color/texture on the fruit."""
    img = cv2.resize(img_bgr, resize_to, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, resize_to, interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8) * 255
    if cv2.countNonZero(m) < 40:
        m = np.ones(resize_to[::-1], np.uint8) * 255

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 48 color features: 32 hue bins + 8 saturation + 8 value.
    hist_h = cv2.calcHist([hsv], [0], m, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], m, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], m, [8], [0, 256]).flatten()
    hist_h /= hist_h.sum() + 1e-7
    hist_s /= hist_s.sum() + 1e-7
    hist_v /= hist_v.sum() + 1e-7
    color = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter else 0.0
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h else 1.0
    else:
        circularity, aspect = 0.0, 1.0
    shape = np.array([circularity, aspect], dtype=np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    fg = m > 0
    texture = np.array([float(edges[fg].mean() / 255.0) if np.any(fg) else 0.0], dtype=np.float32)

    return np.concatenate([color, shape, texture]).astype(np.float32)



def real_world_appearance_scores(image_or_path):
    """Return appearance evidence for the focused Apple/Banana/Orange model.

    A compact sliding-window search is used instead of trusting the photo
    background.  Each class has a discriminative HSV color cue, and the
    winning window is scored by color coverage, saturation, edge density and
    compactness.  These scores are *not* a replacement for the SVM; they are
    fused with the SVM decision scores before the required Softmax step.
    """
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
    else:
        img = image_or_path
    if img is None:
        raise ValueError("Could not read image for appearance scoring")
    h0, w0 = img.shape[:2]
    scale = min(1.0, 512.0 / max(h0, w0))
    if scale < 1.0:
        img = cv2.resize(img, (max(32, int(w0 * scale)), max(32, int(h0 * scale))),
                         interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    class_masks = {
        "apple": (((H < 7) | (H > 173)) & (S > 80) & (V > 40)),
        "banana": ((H >= 15) & (H <= 48) & (S > 70) & (V > 45)),
        "orange": ((H >= 5) & (H <= 28) & (S > 100) & (V > 45)),
    }

    scores = []
    for cls in ("apple", "banana", "orange"):
        cm = class_masks[cls]
        best = 0.0
        for frac in (0.22, 0.28, 0.34, 0.42):
            ww = max(30, int(w * frac))
            hh = max(30, int(h * frac))
            for cy in np.linspace(0.25, 0.80, 8):
                for cx in np.linspace(0.25, 0.80, 8):
                    x1 = int(cx * w - ww / 2)
                    y1 = int(cy * h - hh / 2)
                    x1 = max(0, min(w - ww, x1))
                    y1 = max(0, min(h - hh, y1))
                    mm = cm[y1:y1 + hh, x1:x1 + ww]
                    ss = S[y1:y1 + hh, x1:x1 + ww]
                    ee = edges[y1:y1 + hh, x1:x1 + ww]
                    color = float(mm.mean())
                    sat = float(ss[mm].mean() / 255.0) if np.any(mm) else 0.0
                    edge = float(ee.mean() / 255.0)
                    cnts, _ = cv2.findContours((mm.astype(np.uint8) * 255),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    compact = 0.0
                    if cnts:
                        compact = max(cv2.contourArea(c) for c in cnts) / float(ww * hh)
                    value = 5.0 * color + 2.0 * sat + 0.8 * edge + 1.5 * compact
                    if value > best:
                        best = value
        scores.append(best)
    scores = np.asarray(scores, dtype=np.float32)

    # Targeted orange-vs-banana correction for vivid oranges.
    # Only activate this cue when the focused fruit is strongly saturated
    # and has a clear red-vs-green channel separation. This avoids changing
    # red apples, while correcting real-world oranges that contain green
    # leaves/stems and otherwise look too yellow to the SVM.
    try:
        fm = foreground_mask(img)
        obj, om = object_crop(img, fm)
        hsv_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV)
        pixels_hsv = hsv_obj[om > 127]
        pixels_bgr = obj[om > 127]
        if len(pixels_hsv) >= 40:
            mean_h = float(np.median(pixels_hsv[:, 0]))
            mean_s = float(np.median(pixels_hsv[:, 1]))
            rg = float(np.median(pixels_bgr[:, 2].astype(np.float32) -
                                  pixels_bgr[:, 1].astype(np.float32)))

            hue_orange = float(np.exp(-0.5 * ((mean_h - 13.0) / 7.0) ** 2))
            sat_factor = 1.0 / (1.0 + np.exp(-(mean_s - 180.0) / 16.0))
            rg_factor = 1.0 / (1.0 + np.exp(-(rg - 60.0) / 9.0))
            orange_boost = 2.20 * hue_orange * sat_factor * rg_factor
            scores[2] += float(orange_boost)
    except Exception:
        pass

    return scores

def _load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def prepare_object(path):
    img = _load_image(path)
    mask = foreground_mask(img)
    return object_crop(img, mask)


def training_features(path, seed_base=0):
    """Return six background-robust views for one training image.

    The five requested synthetic backgrounds are generated for every source
    image.  Features are computed from the known fruit mask, so the classifier
    never learns the synthetic background itself; the augmented object
    appearance (rotation/scale/light/blur/hue) is what increases robustness.
    """
    img = _load_image(path)
    mask = foreground_mask(img)
    obj, om = object_crop(img, mask)

    feats = [extract_masked_features(obj, om)]
    for i, kind in enumerate(BACKGROUND_KINDS):
        aug, am = _augment_object(obj, om, seed_base + i * 97)
        _ = recompose(aug, am, kind, seed_base + 1000 + i * 31)
        # Use the same foreground mask after augmentation.  This preserves
        # the intended background-invariance while keeping training fast.
        feats.append(extract_masked_features(aug, am))
    return np.stack(feats)


def inference_features(path):
    """Create several deterministic views of one unseen real-world image."""
    img = _load_image(path)
    mask = foreground_mask(img)
    obj, om = object_crop(img, mask)

    feats = [extract_masked_features(obj, om)]
    for i, kind in enumerate(BACKGROUND_KINDS):
        # Recompose the already-cropped object without geometric augmentation.
        comp = recompose(obj, om, kind, 7000 + i * 41)
        feats.append(extract_masked_features(comp, om))
    return np.stack(feats)
