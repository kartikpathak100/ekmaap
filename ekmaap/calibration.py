"""Scale and perspective from a printed ArUco grading mat.

The mat is an A4 sheet (landscape) with four ArUco markers in its corners
(``scripts/make_mat.py`` prints it). Each marker has 4 corners at known
millimetre positions, so 4-16 point pairs give a homography from the photo to
the flat mat. Warping the photo with it gives a top-down image where one pixel
is exactly ``1 / px_per_mm`` millimetres, even when the phone is tilted.

A single loose marker (the "card" mode) also works, but everything far from the
card is extrapolated, so the result is flagged as lower quality.

Known physical limit: the widest part of an onion sits above the mat (about
half its diameter), so it looks slightly larger than it is. The trained size
calibration (``measure.SizeCalibration``) corrects that bias from ruler data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass(frozen=True)
class MatSpec:
    width_mm: float = 297.0
    height_mm: float = 210.0
    marker_mm: float = 40.0
    # top-left corner of each marker on the mat, in mm
    positions: dict = field(default_factory=lambda: {0: (10.0, 10.0), 1: (247.0, 10.0), 2: (247.0, 160.0), 3: (10.0, 160.0)})
    # a loose "size card" marker id and side (used only when no mat marker is seen)
    card_id: int = 10
    card_mm: float = 50.0

    def marker_corners_mm(self, marker_id: int) -> np.ndarray:
        x, y = self.positions[marker_id]
        s = self.marker_mm
        return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], np.float64)


DEFAULT_MAT = MatSpec()


@dataclass
class Calibration:
    mode: str                      # "mat" | "card"
    px_per_mm: float               # scale of the rectified image
    H: np.ndarray                  # 3x3: original image pixels -> rectified pixels
    size: tuple                    # (width, height) of the rectified image in px
    origin_mm: tuple               # mm coordinate of rectified pixel (0, 0)
    markers: list                  # ids used
    reproj_mm: float               # mean corner error after fitting, mm
    quality: str                   # "good" | "fair" | "poor"

    def to_json(self) -> dict:
        d = asdict(self)
        d["H"] = np.asarray(self.H).round(8).tolist()
        d["size"] = list(self.size)
        d["origin_mm"] = list(self.origin_mm)
        return d

    def to_original(self, pts_rect: np.ndarray) -> np.ndarray:
        """Map rectified pixel coordinates back to the original photo."""
        pts = np.asarray(pts_rect, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def to_rect(self, pts_img: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_img, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)


def detect_markers(image_bgr: np.ndarray) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def calibrate(image_bgr: np.ndarray, spec: MatSpec = DEFAULT_MAT, px_per_mm: float = 3.0,
              max_card_extent_mm: float = 420.0) -> Calibration | None:
    """Find the mat (or a loose card) and return the photo -> top-down transform. None if nothing found."""
    found = detect_markers(image_bgr)
    mat_ids = [i for i in found if i in spec.positions]
    S = np.diag([px_per_mm, px_per_mm, 1.0])
    if mat_ids:
        img_pts = np.vstack([found[i] for i in mat_ids])
        mm_pts = np.vstack([spec.marker_corners_mm(i) for i in mat_ids])
        if len(mat_ids) >= 2:
            H_mm, _ = cv2.findHomography(img_pts, mm_pts, cv2.RANSAC, 2.0)
        else:
            H_mm, _ = cv2.findHomography(img_pts, mm_pts, 0)
        if H_mm is None:
            return None
        err = _reproj(H_mm, img_pts, mm_pts)
        size = (int(round(spec.width_mm * px_per_mm)), int(round(spec.height_mm * px_per_mm)))
        quality = "good" if len(mat_ids) >= 3 and err < 1.0 else "fair" if len(mat_ids) >= 2 and err < 2.0 else "poor"
        return Calibration("mat", px_per_mm, S @ H_mm, size, (0.0, 0.0), sorted(mat_ids), float(err), quality)
    if spec.card_id in found:
        s = spec.card_mm
        mm_pts = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float64)
        H_mm, _ = cv2.findHomography(found[spec.card_id], mm_pts, 0)
        if H_mm is None:
            return None
        h, w = image_bgr.shape[:2]
        corners = cv2.perspectiveTransform(np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], np.float64), H_mm).reshape(-1, 2)
        lo = np.maximum(corners.min(0), -max_card_extent_mm / 2)
        hi = np.minimum(corners.max(0), max_card_extent_mm / 2 + s)
        T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], np.float64)
        size = (int(round((hi[0] - lo[0]) * px_per_mm)), int(round((hi[1] - lo[1]) * px_per_mm)))
        err = _reproj(H_mm, found[spec.card_id], mm_pts)
        return Calibration("card", px_per_mm, S @ T @ H_mm, size, (float(lo[0]), float(lo[1])), [spec.card_id], float(err), "poor")
    return None


def _reproj(H: np.ndarray, img_pts: np.ndarray, mm_pts: np.ndarray) -> float:
    proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.linalg.norm(proj - mm_pts, axis=1).mean())


def rectify(image: np.ndarray, cal: Calibration, interpolation=cv2.INTER_LINEAR, border=None) -> np.ndarray:
    if border is None:
        border = (0, 0, 0) if image.ndim == 3 else 0
    return cv2.warpPerspective(image, cal.H, cal.size, flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def valid_region(image_shape: tuple, cal: Calibration) -> np.ndarray:
    """Rectified pixels that came from inside the photo (warping leaves empty borders)."""
    h, w = image_shape[:2]
    ones = np.full((h, w), 255, np.uint8)
    return rectify(ones, cal, cv2.INTER_NEAREST) > 0


def marker_exclusion(cal: Calibration, spec: MatSpec = DEFAULT_MAT, margin_mm: float = 4.0) -> np.ndarray:
    """Mask of rectified pixels covered by markers (plus a margin) - never an onion."""
    W, H = cal.size
    m = np.zeros((H, W), bool)
    p = cal.px_per_mm
    ox, oy = cal.origin_mm
    boxes = []
    if cal.mode == "mat":
        boxes = [(x, y, spec.marker_mm) for (x, y) in spec.positions.values()]
    else:
        boxes = [(0.0, 0.0, spec.card_mm)]
    for x, y, s in boxes:
        x0 = int(max(0, (x - ox - margin_mm) * p)); y0 = int(max(0, (y - oy - margin_mm) * p))
        x1 = int(min(W, (x - ox + s + margin_mm) * p)); y1 = int(min(H, (y - oy + s + margin_mm) * p))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m
