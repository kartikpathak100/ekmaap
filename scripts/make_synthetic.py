"""Write a SYNTHETIC labelled dataset (images/, annotations.json, ruler.csv) to try the whole flow.

    python scripts/make_synthetic.py --out data/synthetic --photos 24

Only for checking that the scripts run. Scores on it say nothing about real onions.
"""
import argparse

from _common import *  # noqa: F401,F403
from ekmaap.synth import make_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--photos", type=int, default=24)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
coco = make_dataset(a.out, a.photos, a.seed)
print(f"wrote {len(coco['images'])} photos, {len(coco['annotations'])} onions to {a.out}")
