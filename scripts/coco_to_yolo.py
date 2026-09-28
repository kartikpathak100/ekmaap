"""Convert COCO labels to a YOLO-seg dataset of RECTIFIED photos (for the optional GPU model).

    python scripts/coco_to_yolo.py --coco data/real/annotations.json --images data/real/images \
        --out data/yolo [--classes onion|conditions]

Photos are warped onto the mat's millimetre grid first (3 px/mm), exactly as the app does before
segmenting, so the model always sees onions at true scale. Photos without a visible mat are skipped.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from _common import *  # noqa: F401,F403
from ekmaap import CONDITIONS
from ekmaap.dataset import iter_samples

ap = argparse.ArgumentParser()
ap.add_argument("--coco", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--classes", choices=["onion", "conditions"], default="onion")
a = ap.parse_args()

out = Path(a.out)
names = ["onion"] if a.classes == "onion" else list(CONDITIONS)
counts = {}
for s in iter_samples(a.coco, a.images):
    if s.cal is None:
        print("skip (no mat):", s.file_name); continue
    (out / "images" / s.split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / s.split).mkdir(parents=True, exist_ok=True)
    stem = Path(s.file_name).stem
    rect = s.rect.copy()
    rect[~s.usable] = 0
    cv2.imwrite(str(out / "images" / s.split / f"{stem}.jpg"), rect, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = rect.shape[:2]
    lines = []
    for k, cond in enumerate(s.conditions, 1):
        m = (s.instances == k).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
        if len(c) < 3:
            continue
        cls = 0 if a.classes == "onion" else CONDITIONS.index(cond)
        lines.append(" ".join([str(cls)] + [f"{v:.5f}" for xy in c / [w, h] for v in xy]))
    (out / "labels" / s.split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    counts[s.split] = counts.get(s.split, 0) + 1
(out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                               + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
print("photos per split:", counts, "->", out / "data.yaml")
