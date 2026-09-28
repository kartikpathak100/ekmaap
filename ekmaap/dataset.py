"""Load labelled photos (COCO + images folder) into the rectified, millimetre-scaled space the models use."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import coco as coco_mod
from .calibration import DEFAULT_MAT
from .pipeline import prepare


@dataclass
class Sample:
    file_name: str
    image_id: int
    split: str
    image: np.ndarray            # original (possibly downscaled) photo
    scale: float                 # downscale factor applied to the original
    cal: object | None           # Calibration or None (no mat found -> unusable for size/seg training)
    rect: np.ndarray | None
    usable: np.ndarray | None    # rectified pixels inside the photo and not on a marker
    instances: np.ndarray | None # rectified instance labels, k = k-th entry of `conditions`
    conditions: list
    polygons: list               # original-photo polygons (full resolution)
    ruler: list                  # [(k, ruler_mm)] - k indexes `conditions` (1-based)
    ppm: float | None


def make_sample(file_name: str, image_id, split: str, img: np.ndarray, polys: list, conditions: list,
                ruler_points=(), px_per_mm: float = 3.0, spec=DEFAULT_MAT) -> Sample:
    """Build one Sample from an image, its labelled polygons (original pixels) and ruler points [(x, y, mm)]."""
    small, cal, rect, valid, exclude, ppm, scale = prepare(img, spec, px_per_mm)
    labels = usable = None
    if cal is not None:
        H = cal.H @ np.diag([scale, scale, 1.0])
        labels = coco_mod.rasterize(polys, rect.shape, H)
        usable = valid & ~exclude
    rl = []
    for (x, y, mm) in ruler_points:
        for k, p in enumerate(polys, 1):
            if cv2.pointPolygonTest(np.asarray(p, np.float32), (float(x), float(y)), False) >= 0:
                rl.append((k, mm)); break
    return Sample(file_name, image_id, split, small, scale, cal, rect, usable, labels, list(conditions),
                  [np.asarray(p, np.float64) for p in polys], rl, ppm)


def iter_samples(coco_path, images_dir, splits=None, ruler_csv=None, px_per_mm: float = 3.0, spec=DEFAULT_MAT):
    coco = coco_mod.load(coco_path)
    split_of = coco_mod.assign_splits(coco)
    ruler = coco_mod.load_ruler(ruler_csv) if ruler_csv else {}
    for im in coco["images"]:
        split = im.get("split") or split_of[im["id"]]
        if split == "unassigned":
            split = split_of[im["id"]]
        if splits and split not in splits:
            continue
        path = Path(images_dir) / im["file_name"]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        inst = coco_mod.instances(coco, im["id"])
        yield make_sample(im["file_name"], im["id"], split, img, [i["polygon"] for i in inst],
                          [i["condition"] for i in inst], ruler.get(im["file_name"], []), px_per_mm, spec)
