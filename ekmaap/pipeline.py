"""Photo -> analysis. Used by the API and by the evaluation script."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import calibration as cal_mod
from .classifier import HeuristicClassifier
from .coco import mask_to_polygon
from .features import onion_features
from .measure import SizeCalibration, measure_mask
from .rules import RuleSet, summarise
from .segmentation import ClassicalSegmenter


@dataclass
class Models:
    segmenter: object = field(default_factory=ClassicalSegmenter)
    classifier: object = field(default_factory=HeuristicClassifier)
    size_cal: SizeCalibration | None = None
    versions: dict = field(default_factory=dict)   # e.g. {"segmenter": "pf-2026-10-01", ...}


def decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a readable image")
    return img


def prepare(img: np.ndarray, spec=cal_mod.DEFAULT_MAT, px_per_mm: float = 3.0, max_side: int = 2400):
    """Calibrate and rectify. Returns (small_img, calibration|None, rect, valid, exclude, ppm, scale)."""
    scale = min(1.0, max_side / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    cal = cal_mod.calibrate(small, spec, px_per_mm)
    if cal is None:
        return small, None, small, None, None, None, scale
    rect = cal_mod.rectify(small, cal)
    return small, cal, rect, cal_mod.valid_region(small.shape, cal), cal_mod.marker_exclusion(cal, spec), px_per_mm, scale


def analyse(img: np.ndarray, models: Models, rules: RuleSet, spec=cal_mod.DEFAULT_MAT, px_per_mm: float = 3.0) -> dict:
    t0 = time.perf_counter()
    small, cal, rect, valid, exclude, ppm, scale = prepare(img, spec, px_per_mm)
    if cal is None:
        # No mat/card: we can still find onions, but not sizes. Classical only works on a plain sheet.
        ppm_guess = max(small.shape[:2]) / 300.0
        labels = models.segmenter.segment(rect, ppm_guess, None, None)
    else:
        labels = models.segmenter.segment(rect, ppm, valid, exclude)
    t_seg = time.perf_counter()
    onions = []
    H, W = labels.shape
    for k in range(1, int(labels.max()) + 1):
        m = labels == k
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        edge = xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1
        if valid is not None:
            ring = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            edge = edge or bool((ring & ~valid).any())
        meas = measure_mask(m, ppm, models.size_cal) if ppm else measure_mask(m, 1.0)
        roundness = meas.width_mm / meas.diameter_mm if meas and meas.diameter_mm else 1.0
        feats = onion_features(rect, m, meas.solidity if meas else 1.0, roundness)
        pred = models.classifier.predict(feats, rules.review_below_confidence)
        review = pred.review or edge
        reasons = list(pred.reasons) + (["cut off by the photo edge"] if edge else []) + ([] if ppm else ["no mat or card found: size unknown"])
        d = meas.diameter_mm if (meas and ppm) else None
        grade, label = rules.grade(pred.condition, d, review)
        poly_rect = mask_to_polygon(m)
        if poly_rect is None:
            continue
        poly_img = (cal.to_original(poly_rect) if cal else poly_rect) / scale
        onions.append({
            "idx": len(onions) + 1,
            "polygon": np.round(poly_img, 1).tolist(),               # original photo pixels
            "centroid": np.round(poly_img.mean(0), 1).tolist(),
            "diameter_mm": round(d, 1) if d is not None else None,
            "raw_diameter_mm": round(meas.raw_diameter_mm, 1) if (meas and ppm) else None,
            "width_mm": round(meas.width_mm, 1) if (meas and ppm) else None,
            "condition": pred.condition, "proba": {k2: round(v, 3) for k2, v in pred.proba.items()},
            "grade": grade, "label": label, "review": review, "reasons": reasons,
            "features": {k2: round(v, 4) for k2, v in feats.items()},
        })
    summary = summarise([o["grade"] for o in onions])
    return {
        "calibration": cal.to_json() if cal else None,
        "scale_found": cal is not None,
        "image_size": [int(img.shape[1]), int(img.shape[0])],
        "onions": onions,
        "summary": summary,
        "rule_set": rules.model_dump(),
        "models": {"segmenter": models.segmenter.to_json(), "classifier": models.classifier.to_json(),
                   "size_calibration": models.size_cal.__dict__ if models.size_cal else None, **models.versions},
        "timing_ms": {"segment": round(1000 * (t_seg - t0)), "total": round(1000 * (time.perf_counter() - t0))},
    }


CODE = {"Grade A": "A", "Undersized": "U", "Sprouted": "S", "Damaged": "D", "Rotten": "R", "Needs review": "?"}
GRADE_BGR = {"Grade A": (69, 123, 30), "URS": (14, 124, 199), "Reject": (30, 38, 179), "Review": (181, 63, 106)}


def annotate(img: np.ndarray, analysis: dict) -> np.ndarray:
    out = img.copy()
    t = max(2, int(max(img.shape[:2]) / 600))
    fs = max(0.5, max(img.shape[:2]) / 1800)
    for o in analysis["onions"]:
        col = GRADE_BGR[o["grade"]]
        cv2.polylines(out, [np.int32(o["polygon"])], True, col, t, cv2.LINE_AA)
        txt = f"{o['idx']} {CODE.get(o['label'], '?')}" + (f" {o['diameter_mm']:.0f}mm" if o["diameter_mm"] else "")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cx, cy = map(int, o["centroid"])
        cv2.rectangle(out, (cx - tw // 2 - 4, cy - th - 4), (cx + tw // 2 + 4, cy + 6), col, -1)
        cv2.putText(out, txt, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return out
