"""Run the current models on unlabelled photos and write a COCO file to CORRECT in CVAT.

    python scripts/prelabel.py --images data/new/images --out data/new/prelabels.json \
        [--segmenter models/segmenter_pf.joblib --classifier models/classifier.joblib]

In CVAT: create a task with the photos, then Actions -> Upload annotations -> "COCO 1.0" -> this file.
Fix outlines and conditions, then export as COCO 1.0. Correcting is much faster than drawing.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from _common import load_models, load_rules, write_json
from ekmaap.coco import add_annotation, add_image, new_coco
from ekmaap.pipeline import analyse

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--segmenter")
ap.add_argument("--classifier")
a = ap.parse_args()

models, rules = load_models(a.segmenter, a.classifier, None), load_rules()
coco = new_coco()
files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
for p in files:
    img = cv2.imread(str(p))
    res = analyse(img, models, rules)
    iid = add_image(coco, p.name, img.shape[1], img.shape[0])
    for o in res["onions"]:                      # polygons are in original photo pixels
        add_annotation(coco, iid, np.asarray(o["polygon"]), o["condition"])
    print(f"{p.name}: {len(res['onions'])} onions, mat found: {res['scale_found']}")
write_json(a.out, coco)
print("wrote", a.out)
