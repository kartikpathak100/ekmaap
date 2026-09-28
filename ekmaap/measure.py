"""Widest diameter (mm) of an onion from its mask in the rectified image.

1. Trim thin parts (neck, root tuft, sprout) with a morphological opening sized to the onion.
2. Fit an ellipse with the same second moments as the trimmed body; its long axis is the diameter.
   If the outline is dented (a cut or split, solidity < 0.95), fit the ellipse to the convex hull,
   because a cut removes flesh but not diameter.
3. Optionally correct systematic bias with a trained ``SizeCalibration`` (fitted from ruler data):
   the widest part of an onion sits above the mat, so photos read slightly large.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Measurement:
    diameter_mm: float
    width_mm: float
    raw_diameter_mm: float   # before size calibration
    feret_mm: float          # plain max caliper distance (for reference)
    solidity: float
    area_mm2: float
    method: str


def _moment_axes(m: dict) -> tuple[float, float]:
    if m["m00"] <= 0:
        return 0.0, 0.0
    cxx, cyy, cxy = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = np.sqrt(max(0.0, tr * tr / 4 - det))
    return 4 * np.sqrt(tr / 2 + disc), 4 * np.sqrt(max(0.0, tr / 2 - disc))


def measure_mask(mask: np.ndarray, ppm: float, size_cal: "SizeCalibration | None" = None) -> Measurement | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    pad = 4
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    sub = mask[y0:ys.max() + pad + 1, x0:xs.max() + pad + 1].astype(np.uint8)
    area = float(sub.sum())
    eq_r = np.sqrt(area / np.pi)
    r = max(1, int(round(0.15 * eq_r)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    body = cv2.morphologyEx(np.pad(sub, r + 1), cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(body)
    if n <= 1:
        body = np.pad(sub, r + 1)
    else:
        body = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    solidity = float(body.sum() / max(1.0, cv2.contourArea(hull) + 0.5 * cv2.arcLength(hull, True)))
    if solidity < 0.95:
        major, minor = _moment_axes(cv2.moments(hull.astype(np.float32)))
        method = "hull-ellipse"
    else:
        major, minor = _moment_axes(cv2.moments(body, binaryImage=True))
        method = "ellipse"
    hp = hull.reshape(-1, 2).astype(np.float64)
    feret = max((np.linalg.norm(hp[i] - hp[j]) for i in range(len(hp)) for j in range(i + 1, len(hp))), default=0.0)
    raw = major / ppm
    d = size_cal.apply(raw) if size_cal else raw
    return Measurement(float(d), float(minor / ppm), float(raw), float(feret / ppm), solidity, area / ppm ** 2, method)


@dataclass
class SizeCalibration:
    """true_mm = slope * measured_mm + intercept, fitted on ruler-measured onions.

    Robust: gross mismatches (e.g. two touching onions measured as one) are segmentation failures,
    not scale errors, so points far from the fit (> 3.5 robust SD) are excluded and counted.
    """
    slope: float = 1.0
    intercept: float = 0.0
    n: int = 0
    n_outliers: int = 0
    mae_before: float | None = None       # inliers, no correction
    mae_after_cv: float | None = None     # inliers, correction fitted without that photo
    helps: bool = True

    def apply(self, mm: float) -> float:
        return self.slope * mm + self.intercept

    @staticmethod
    def _robust(measured, ruler):
        keep = np.ones(len(measured), bool)
        for _ in range(4):
            A = np.stack([measured[keep], np.ones(keep.sum())], 1)
            slope, icpt = np.linalg.lstsq(A, ruler[keep], rcond=None)[0]
            r = ruler - (slope * measured + icpt)
            mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) or 0.5
            new = np.abs(r - np.median(r[keep])) <= max(3.5 * mad, 1.0)
            if (new == keep).all():
                break
            keep = new
        return float(slope), float(icpt), keep

    @classmethod
    def fit(cls, measured: np.ndarray, ruler: np.ndarray, groups: np.ndarray | None = None) -> "SizeCalibration":
        measured, ruler = np.asarray(measured, float), np.asarray(ruler, float)
        if len(measured) < 5:
            raise ValueError("Need at least 5 ruler-measured onions to fit a size calibration")
        slope, icpt, keep = cls._robust(measured, ruler)
        g = np.asarray(groups) if groups is not None else np.arange(len(measured))
        errs = []
        for gv in np.unique(g):                     # leave one photo out
            tr, te = (g != gv) & keep, (g == gv) & keep
            if tr.sum() < 3 or not te.any():
                continue
            s_, b_, _ = cls._robust(measured[tr], ruler[tr])
            errs.extend(np.abs(s_ * measured[te] + b_ - ruler[te]))
        before = float(np.mean(np.abs(measured[keep] - ruler[keep])))
        after = float(np.mean(errs)) if errs else None
        return cls(slope, icpt, int(keep.sum()), int((~keep).sum()), before, after, after is not None and after < before)

    def save(self, path):
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path) -> "SizeCalibration":
        return cls(**json.loads(Path(path).read_text()))
