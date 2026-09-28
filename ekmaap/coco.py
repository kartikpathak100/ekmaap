"""COCO instance-segmentation files: the exchange format with labelling tools.

Convention used everywhere in this repo:
  * one category per condition: sound, sprouted, damaged, rotten (every tool supports categories;
    not every tool exports attributes)
  * one polygon per onion, in ORIGINAL photo pixels
  * optional ``ruler.csv`` next to it: file_name, x, y, ruler_mm - a point inside an onion and its
    diameter measured with a ruler or caliper (used for size calibration and size evaluation)

CVAT ("COCO 1.0") and Label Studio (COCO export) both read and write this.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import CONDITIONS


def new_coco() -> dict:
    return {"info": {"description": "EkMaap onion annotations"}, "images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": c, "supercategory": "onion"} for i, c in enumerate(CONDITIONS)]}


def add_image(coco: dict, file_name: str, width: int, height: int, **extra) -> int:
    image_id = len(coco["images"]) + 1
    coco["images"].append({"id": image_id, "file_name": file_name, "width": int(width), "height": int(height), **extra})
    return image_id


def add_annotation(coco: dict, image_id: int, polygon: np.ndarray, condition: str, **extra) -> int:
    poly = np.asarray(polygon, np.float64).reshape(-1, 2)
    cat = {c["name"]: c["id"] for c in coco["categories"]}[condition]
    x0, y0 = poly.min(0); x1, y1 = poly.max(0)
    ann_id = len(coco["annotations"]) + 1
    coco["annotations"].append({"id": ann_id, "image_id": image_id, "category_id": cat, "iscrowd": 0,
                                "segmentation": [poly.round(1).ravel().tolist()], "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                                "area": float(cv2.contourArea(poly.astype(np.float32))), **extra})
    return ann_id


def load(path: str | Path) -> dict:
    coco = json.loads(Path(path).read_text())
    names = {c["id"]: c["name"].strip().lower() for c in coco["categories"]}
    unknown = sorted(set(names.values()) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown categories {unknown}; expected {CONDITIONS}")
    coco["_cat_name"] = names
    return coco


def instances(coco: dict, image_id: int) -> list[dict]:
    """[{polygon: (N,2) array, condition: str, id}] for one image (multi-part polygons: largest part)."""
    out = []
    for a in coco["annotations"]:
        if a["image_id"] != image_id or not a.get("segmentation") or isinstance(a["segmentation"], dict):
            continue
        parts = [np.asarray(s, np.float64).reshape(-1, 2) for s in a["segmentation"] if len(s) >= 6]
        if not parts:
            continue
        poly = max(parts, key=lambda p: cv2.contourArea(p.astype(np.float32)))
        out.append({"id": a["id"], "polygon": poly, "condition": coco["_cat_name"][a["category_id"]]})
    return out


def rasterize(polygons: list[np.ndarray], shape: tuple, H: np.ndarray | None = None) -> np.ndarray:
    """Instance label image (0 = background, k = k-th polygon). Optional homography applied to the polygons."""
    lab = np.zeros(shape[:2], np.int32)
    for k, p in enumerate(polygons, start=1):
        q = np.asarray(p, np.float64).reshape(-1, 1, 2)
        if H is not None:
            q = cv2.perspectiveTransform(q, H)
        cv2.fillPoly(lab, [np.round(q.reshape(-1, 2)).astype(np.int32)], k)
    return lab


def mask_to_polygon(mask: np.ndarray, epsilon: float = 1.0) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2).astype(np.float64)
    return c if len(c) >= 3 else None


def assign_splits(coco: dict, fractions=(0.7, 0.15, 0.15), key: str = "split_group") -> dict[int, str]:
    """Split by PHOTO (or by a "split_group" such as a capture day, when images carry one), never by onion.

    Onions in one photo share light, background and camera, so splitting onions would leak and
    inflate scores. The split is a stable hash of the key, so adding photos never reshuffles old ones.
    """
    out = {}
    for im in coco["images"]:
        k = str(im.get(key) or im["file_name"])
        h = int(hashlib.sha1(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out[im["id"]] = "train" if h < fractions[0] else "val" if h < fractions[0] + fractions[1] else "test"
    return out


def load_ruler(path: str | Path) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list] = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["file_name"], []).append((float(r["x"]), float(r["y"]), float(r["ruler_mm"])))
    return out
