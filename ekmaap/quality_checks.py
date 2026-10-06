"""Photo quality gate: tell the officer BEFORE analysis if the photo is too poor to trust.

Checks sharpness (variance of the Laplacian), brightness, glare (share of near-white pixels) and
resolution. Thresholds are starting values to be tuned on real photos; they are parameters, not constants
hidden in the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityLimits:
    min_side_px: int = 800
    min_sharpness: float = 60.0      # variance of Laplacian on a 800 px-wide grey image
    dark_below: float = 60.0         # mean grey level 0-255
    bright_above: float = 215.0
    max_glare_share: float = 0.08    # share of pixels >= 250 in all channels


@dataclass
class QualityReport:
    ok: bool
    metrics: dict
    warnings: list[str] = field(default_factory=list)


def check(img_bgr: np.ndarray, limits: QualityLimits | None = None) -> QualityReport:
    lim = limits or QualityLimits()
    h, w = img_bgr.shape[:2]
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if w != 800:                                    # make sharpness comparable between phone resolutions
        grey = cv2.resize(grey, (800, max(1, round(h * 800 / w))), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    mean = float(np.mean(grey))
    glare = float(np.mean(np.all(img_bgr >= 250, axis=2)))
    m = {"width_px": w, "height_px": h, "sharpness": round(sharp, 1),
         "mean_brightness": round(mean, 1), "glare_share": round(glare, 4)}
    warn: list[str] = []
    if min(h, w) < lim.min_side_px:
        warn.append(f"Photo is small ({w}x{h}). Move closer or use the phone's normal camera, not a zoomed or reduced copy.")
    if sharp < lim.min_sharpness:
        warn.append("Photo looks blurred. Hold the phone steady and tap the screen to focus before taking it.")
    if mean < lim.dark_below:
        warn.append("Photo is too dark. Move to better light or switch on the room light.")
    if mean > lim.bright_above:
        warn.append("Photo is too bright. Avoid direct sunlight on the mat.")
    if glare > lim.max_glare_share:
        warn.append("Strong glare or shiny patches. Change the angle so lamps and windows are not reflected.")
    return QualityReport(ok=not warn, metrics=m, warnings=warn)
