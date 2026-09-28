"""Onion instance segmentation on the rectified (top-down, millimetre-scaled) image.

Three interchangeable segmenters, all returning an instance label image
(0 = not an onion, k = onion number k):

* ``ClassicalSegmenter``   - no training. Colour distance from the mat paper + distance-transform
                              splitting. This is the old demo logic; it breaks on real photos with
                              glare, shadows and busy backgrounds.
* ``PixelForestSegmenter`` - TRAINABLE on a CPU in minutes. A random forest labels every pixel as
                              background / onion interior / onion boundary from colour and texture
                              features (the approach of the ilastik tool). Boundary pixels between
                              touching onions let watershed split them. Needs ~20+ labelled photos.
* ``YoloSegmenter``         - optional, needs a GPU to train (``scripts/train_yolo_seg.py``). Best
                              when you have a few hundred labelled photos.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

FEATURE_VERSION = 1


# ----------------------------------------------------------------------------- pixel features
def pixel_features(img_bgr: np.ndarray) -> np.ndarray:
    """(H, W, 17) float32: Lab at 4 blur scales, gradient, Laplacian, local contrast, saturation."""
    f32 = img_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f32, cv2.COLOR_BGR2LAB)          # L 0..100, a/b about -127..127
    feats = []
    for s in (0, 1, 2, 4):
        feats.append(lab if s == 0 else cv2.GaussianBlur(lab, (0, 0), s))
    L = lab[..., 0]
    for s in (1, 2):
        Ls = cv2.GaussianBlur(L, (0, 0), s)
        gx, gy = cv2.Sobel(Ls, cv2.CV_32F, 1, 0), cv2.Sobel(Ls, cv2.CV_32F, 0, 1)
        feats.append(np.sqrt(gx * gx + gy * gy)[..., None])
    feats.append(cv2.Laplacian(cv2.GaussianBlur(L, (0, 0), 2), cv2.CV_32F)[..., None])
    m = cv2.GaussianBlur(L, (0, 0), 2)
    feats.append(np.sqrt(np.maximum(cv2.GaussianBlur(L * L, (0, 0), 2) - m * m, 0))[..., None])
    hsv = cv2.cvtColor(f32, cv2.COLOR_BGR2HSV)
    feats.append(hsv[..., 1:2])
    return np.concatenate(feats, axis=2).astype(np.float32)


# ----------------------------------------------------------------------------- shared helpers
def _clean(fg: np.ndarray, ppm: float) -> np.ndarray:
    r = max(1, int(round(ppm * 0.7)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    return ndi.binary_fill_holes(fg > 0)


def _min_area_px(ppm: float, min_diam_mm: float) -> float:
    return np.pi * (min_diam_mm * ppm / 2) ** 2


def _solidity(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 1.0
    c = max(cnts, key=cv2.contourArea)
    ha = cv2.contourArea(cv2.convexHull(c))
    return float(mask.sum() / ha) if ha > 0 else 1.0


def split_by_distance(comp: np.ndarray, ppm: float) -> np.ndarray | None:
    """Split one blob of touching onions. Returns a label image of the parts or None (keep whole).

    Cores = pixels far from the blob edge, tried at several depths; parts grow back by watershed.
    A split is accepted only if every part is convex (solidity >= 0.9) and, for two parts, the cut
    is short compared with the smaller part - a single onion with a deep cut is not split in two.
    """
    dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    mx = dist.max()
    best = None
    for frac in (0.6, 0.5, 0.4, 0.3):
        n, cores = cv2.connectedComponents((dist >= frac * mx).astype(np.uint8))
        sizes = np.bincount(cores.ravel())
        keep = [i for i in range(1, n) if sizes[i] >= max(12, 0.005 * comp.sum())]
        if len(keep) < 2 or (best is not None and len(keep) <= best.max()):
            continue
        markers = np.zeros_like(cores)
        for j, i in enumerate(keep, 1):
            markers[cores == i] = j
        parts = watershed(-dist, markers, mask=comp)
        ok = True
        eq = []
        for j in range(1, len(keep) + 1):
            pm = parts == j
            if _solidity(pm) < 0.9:
                ok = False
                break
            eq.append(2 * np.sqrt(pm.sum() / np.pi))
        if ok and len(keep) == 2:
            a = parts == 1
            cut = a & ndi.binary_dilation(parts == 2)
            ys, xs = np.nonzero(cut)
            if len(xs) > 1:
                pts = np.stack([xs, ys], 1).astype(np.float32)
                hull = cv2.convexHull(pts).reshape(-1, 2)
                chord = max(np.linalg.norm(hull[i] - hull[j]) for i in range(len(hull)) for j in range(i + 1, len(hull)))
                ok = chord <= 0.75 * min(eq)
        if ok:
            best = parts
    return best


def instances_from_masks(fg: np.ndarray, ppm: float, min_diam_mm: float = 15.0,
                         interior: np.ndarray | None = None) -> np.ndarray:
    """Foreground (and optional learned interior-without-boundary) -> instance label image."""
    fg = _clean(fg, ppm)
    min_area = _min_area_px(ppm, min_diam_mm)
    n, comps = cv2.connectedComponents(fg.astype(np.uint8))
    out = np.zeros(fg.shape, np.int32)
    nxt = 1
    for i in range(1, n):
        comp = comps == i
        area = comp.sum()
        if area < min_area:
            continue
        parts = None
        if interior is not None:
            core = interior & comp
            core = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            m, cl = cv2.connectedComponents(core.astype(np.uint8))
            sizes = np.bincount(cl.ravel())
            seeds = [j for j in range(1, m) if sizes[j] >= 0.1 * min_area]
            if len(seeds) >= 2:
                markers = np.zeros_like(cl)
                for k, j in enumerate(seeds, 1):
                    markers[cl == j] = k
                dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
                parts = watershed(-dist, markers, mask=comp)
            elif _solidity(comp) < 0.95:          # learned boundary missed the contact: fall back to shape
                parts = split_by_distance(comp, ppm)
        elif _solidity(comp) < 0.95:
            parts = split_by_distance(comp, ppm)
        if parts is None:
            out[comp] = nxt; nxt += 1
        else:
            for k in range(1, parts.max() + 1):
                pm = parts == k
                if pm.sum() >= 0.5 * min_area:
                    out[pm] = nxt; nxt += 1
    return out


def _restrict(fg: np.ndarray, valid: np.ndarray | None, exclude: np.ndarray | None) -> np.ndarray:
    if valid is not None:
        fg = fg & valid
    if exclude is not None:
        fg = fg & ~exclude
    return fg


# ----------------------------------------------------------------------------- segmenters
class ClassicalSegmenter:
    kind = "classical"

    def __init__(self, chroma_t: float = 12.0, dark_t: float = 40.0, min_diam_mm: float = 15.0):
        self.chroma_t, self.dark_t, self.min_diam_mm = chroma_t, dark_t, min_diam_mm

    def segment(self, rect: np.ndarray, ppm: float, valid=None, exclude=None) -> np.ndarray:
        lab = cv2.cvtColor(rect.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        region = valid if valid is not None else np.ones(rect.shape[:2], bool)
        if exclude is not None:
            region = region & ~exclude
        bg = np.median(lab[region], axis=0)                 # the mat paper dominates the mat area
        dC = np.hypot(lab[..., 1] - bg[1], lab[..., 2] - bg[2])
        dL = bg[0] - lab[..., 0]
        fg = _restrict((dC > self.chroma_t) | (dL > self.dark_t), valid, exclude)
        return instances_from_masks(fg, ppm, self.min_diam_mm)

    def to_json(self):
        return {"kind": self.kind, "chroma_t": self.chroma_t, "dark_t": self.dark_t}


class PixelForestSegmenter:
    """Per-pixel random forest: 0 = background, 1 = onion interior, 2 = onion boundary."""
    kind = "pixel_forest"

    def __init__(self, n_estimators: int = 60, max_depth: int = 18, work_scale: float = 0.5,
                 boundary_px: int = 2, min_diam_mm: float = 15.0, seed: int = 0):
        from sklearn.ensemble import RandomForestClassifier
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=5,
                                         class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        self.work_scale, self.boundary_px, self.min_diam_mm = work_scale, boundary_px, min_diam_mm
        self.meta: dict = {}

    def _targets(self, inst: np.ndarray) -> np.ndarray:
        t = (inst > 0).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        edge = (cv2.dilate(inst.astype(np.float32), k) != cv2.erode(inst.astype(np.float32), k))
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2 * self.boundary_px + 1,) * 2, np.uint8)) > 0
        t[edge & (inst > 0)] = 2                      # band lies inside the onions: fg = interior + boundary
        return t

    def fit(self, samples, per_image: int = 30000, seed: int = 0):
        """samples: iterable of (rect_bgr, instance_labels, usable_mask) at the same px/mm."""
        rng = np.random.default_rng(seed)
        X, y = [], []
        for rect, inst, usable in samples:
            small = cv2.resize(rect, None, fx=self.work_scale, fy=self.work_scale, interpolation=cv2.INTER_AREA)
            ins = cv2.resize(inst.astype(np.float32), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            use = cv2.resize(usable.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            F = pixel_features(small)
            t = self._targets(ins)
            for cls, share in ((0, 0.4), (1, 0.4), (2, 0.2)):
                idx = np.flatnonzero((t == cls) & use)
                if len(idx):
                    pick = rng.choice(idx, size=min(len(idx), int(per_image * share)), replace=False)
                    X.append(F.reshape(-1, F.shape[2])[pick]); y.append(np.full(len(pick), cls))
        X, y = np.concatenate(X), np.concatenate(y)
        self.rf.fit(X, y)
        self.meta = {"n_pixels": int(len(y)), "class_counts": np.bincount(y, minlength=3).tolist(), "feature_version": FEATURE_VERSION}
        return self

    def probabilities(self, rect: np.ndarray) -> np.ndarray:
        small = cv2.resize(rect, None, fx=self.work_scale, fy=self.work_scale, interpolation=cv2.INTER_AREA)
        F = pixel_features(small)
        p = self.rf.predict_proba(F.reshape(-1, F.shape[2]))
        full = np.zeros((p.shape[0], 3), np.float32)
        full[:, self.rf.classes_.astype(int)] = p
        full = full.reshape(small.shape[0], small.shape[1], 3)
        return cv2.resize(full, (rect.shape[1], rect.shape[0]), interpolation=cv2.INTER_LINEAR)

    def segment(self, rect: np.ndarray, ppm: float, valid=None, exclude=None) -> np.ndarray:
        p = self.probabilities(rect)
        fg = _restrict((p[..., 1] + p[..., 2]) > 0.5, valid, exclude)
        interior = (p[..., 1] > 0.5) & (p[..., 2] < 0.4)
        return instances_from_masks(fg, ppm, self.min_diam_mm, interior=interior)

    def save(self, path):
        joblib.dump({"kind": self.kind, "segmenter": self}, path, compress=3)

    def to_json(self):
        return {"kind": self.kind, "work_scale": self.work_scale, **self.meta}


class YoloSegmenter:
    """Ultralytics YOLO-seg, trained on rectified images (scripts/coco_to_yolo.py + train_yolo_seg.py).

    NOTE: not exercised in this repo's tests - the build machine had no PyTorch. Check it on a
    machine with ``pip install ultralytics`` before relying on it. Ultralytics is AGPL-3.0 licensed.
    """
    kind = "yolo_seg"

    def __init__(self, weights: str | Path, conf: float = 0.35, min_diam_mm: float = 15.0):
        from ultralytics import YOLO  # imported lazily: optional dependency
        self.weights, self.conf, self.min_diam_mm = str(weights), conf, min_diam_mm
        self.model = YOLO(self.weights)

    def segment(self, rect: np.ndarray, ppm: float, valid=None, exclude=None) -> np.ndarray:
        r = self.model.predict(rect, conf=self.conf, retina_masks=True, verbose=False)[0]
        out = np.zeros(rect.shape[:2], np.int32)
        if r.masks is None:
            return out
        order = np.argsort(r.boxes.conf.cpu().numpy())          # paint low confidence first
        k = 1
        for i in order:
            poly = np.asarray(r.masks.xy[i], np.float32)
            if len(poly) < 3:
                continue
            m = np.zeros(rect.shape[:2], np.uint8)
            cv2.fillPoly(m, [poly.round().astype(np.int32)], 1)
            m = _restrict(m > 0, valid, exclude)
            if m.sum() >= 0.5 * _min_area_px(ppm, self.min_diam_mm):
                out[m] = k; k += 1
        return out

    def to_json(self):
        return {"kind": self.kind, "weights": self.weights, "conf": self.conf}


def load_segmenter(path: str | Path | None):
    """Load a saved segmenter. None -> classical baseline; *.pt / *.onnx -> YOLO; *.joblib -> pixel forest."""
    if path is None:
        return ClassicalSegmenter()
    p = Path(path)
    if p.suffix in (".pt", ".onnx"):
        return YoloSegmenter(p)
    obj = joblib.load(p)
    if obj.get("kind") != "pixel_forest":
        raise ValueError(f"{p} is not a segmenter file")
    return obj["segmenter"]
