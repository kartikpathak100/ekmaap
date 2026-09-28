"""Evaluate the full pipeline on the TEST split. These are the numbers you may quote.

    python scripts/evaluate.py --coco ... --images ... --ruler ... \
        --segmenter models/segmenter_pf.joblib --classifier models/classifier.joblib \
        --size-cal models/size_cal.json --out reports/eval_test.json
"""
import argparse
from pathlib import Path

from _common import load_models, load_rules, write_json
from ekmaap.dataset import iter_samples
from ekmaap.evaluate import evaluate, to_markdown

ap = argparse.ArgumentParser()
ap.add_argument("--coco", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--ruler")
ap.add_argument("--segmenter")
ap.add_argument("--classifier")
ap.add_argument("--size-cal")
ap.add_argument("--rules")
ap.add_argument("--split", default="test")
ap.add_argument("--out", required=True)
a = ap.parse_args()

samples = list(iter_samples(a.coco, a.images, splits={a.split}, ruler_csv=a.ruler))
if not samples:
    raise SystemExit(f"No photos in split '{a.split}'")
r = evaluate(samples, load_models(a.segmenter, a.classifier, a.size_cal), load_rules(a.rules))
r["setup"] = {"split": a.split, "segmenter": a.segmenter or "classical", "classifier": a.classifier or "heuristic", "size_cal": a.size_cal}
write_json(a.out, r)
md = to_markdown(r)
Path(a.out).with_suffix(".md").write_text(md + "\n")
print(md)
