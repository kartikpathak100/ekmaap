"""Fit the size correction from ruler measurements (removes the "onions look bigger" bias).

    python scripts/fit_size_calibration.py --coco ... --images ... --ruler data/real/ruler.csv \
        --segmenter models/segmenter_pf.joblib --out models/size_cal.json

For each ruler row the onion containing the point is found in the PIPELINE's own output (same
segmenter as in the field), so the correction matches what the app really measures.
"""
import argparse

import cv2
import numpy as np

from _common import load_models, load_rules
from ekmaap.dataset import iter_samples
from ekmaap.measure import SizeCalibration
from ekmaap.pipeline import analyse

ap = argparse.ArgumentParser()
ap.add_argument("--coco", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--ruler", required=True)
ap.add_argument("--segmenter")
ap.add_argument("--out", required=True)
a = ap.parse_args()

models = load_models(a.segmenter, None, None)
meas, true, groups = [], [], []
for s in iter_samples(a.coco, a.images, splits={"train", "val"}, ruler_csv=a.ruler):
    if not s.ruler or s.cal is None:
        continue
    res = analyse(s.image, models, load_rules())
    for k, mm in s.ruler:
        pt = s.polygons[k - 1].mean(0) * s.scale
        for o in res["onions"]:
            if o["raw_diameter_mm"] and cv2.pointPolygonTest(np.float32(o["polygon"]), tuple(map(float, pt)), False) >= 0:
                meas.append(o["raw_diameter_mm"]); true.append(mm); groups.append(s.file_name)
                break
cal = SizeCalibration.fit(np.array(meas), np.array(true), np.array(groups))
print(f"fit: true = {cal.slope:.4f} x measured + {cal.intercept:+.2f} mm (n={cal.n}, {cal.n_outliers} gross mismatches excluded)")
print(f"mean abs error before: {cal.mae_before:.2f} mm; after, photo held out: "
      f"{cal.mae_after_cv:.2f} mm" if cal.mae_after_cv is not None else "not enough photos to cross-validate")
if not cal.helps:
    raise SystemExit("The correction does not reduce the held-out error - NOT saved. Collect more ruler data.")
cal.save(a.out)
print("saved", a.out)
