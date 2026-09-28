"""Numbers that describe one onion's surface and shape - the classifier's input.

Computed on the rectified image inside the onion's mask. Chosen to be cheap and
explainable: an officer can be told "32% of the surface is very dark".
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

HUE_BINS = 8
LBP_BINS = 10  # uniform LBP with P=8 gives 10 codes

NAMES = (
    ["L_mean", "L_std", "L_p10", "L_p50", "L_p90", "a_mean", "a_std", "b_mean", "b_std",
     "dark_frac", "green_frac", "pale_frac", "brown_frac", "solidity", "roundness", "edge_density"]
    + [f"hue_{i}" for i in range(HUE_BINS)] + [f"lbp_{i}" for i in range(LBP_BINS)]
)


def onion_features(rect_bgr: np.ndarray, mask: np.ndarray, solidity: float, roundness: float) -> dict:
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = rect_bgr[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1].astype(bool)
    f32 = crop.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f32, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0][m], lab[..., 1][m], lab[..., 2][m]
    p10, p50, p75, p90 = np.percentile(L, [10, 50, 75, 90])
    dark = (L < min(28.0, p75 - 18.0))
    green = (a < -6) & (b > 4)
    pale = (L > 60) & (L > p50 + 18) & (np.hypot(a, b) < 30)
    brown = (a > 2) & (b > 15) & (L < p50)                   # dry rot / sun scald tones
    hsv = cv2.cvtColor(f32, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0][m] / 360.0
    sat = hsv[..., 1][m]
    hh = np.histogram(hue, bins=HUE_BINS, range=(0, 1), weights=sat)[0]
    hh = hh / max(hh.sum(), 1e-6)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, "uniform")[m]
    lh = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS))[0].astype(float)
    lh /= max(lh.sum(), 1)
    edges = cv2.Canny(gray, 50, 120) > 0
    inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    vals = [L.mean(), L.std(), p10, p50, p90, a.mean(), a.std(), b.mean(), b.std(),
            dark.mean(), green.mean(), pale.mean(), brown.mean(), solidity, roundness,
            edges[inner].mean() if inner.any() else 0.0] + hh.tolist() + lh.tolist()
    return {k: float(v) for k, v in zip(NAMES, vals)}


def vector(f: dict) -> np.ndarray:
    return np.array([f[k] for k in NAMES], np.float64)
