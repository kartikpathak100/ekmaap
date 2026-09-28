"""Train the pixel-forest onion segmenter on labelled photos (CPU, minutes).

    python scripts/train_segmenter.py --coco data/real/annotations.json --images data/real/images \
        --out models/segmenter_pf.joblib

Uses the TRAIN split (split by photo). Photos where the mat is not found are skipped - they cannot
be put on the millimetre grid. Prints detection scores on the VAL split with the new segmenter.
"""
import argparse
import time

from _common import load_models, load_rules, write_json
from ekmaap.dataset import iter_samples
from ekmaap.evaluate import evaluate, to_markdown
from ekmaap.segmentation import PixelForestSegmenter

ap = argparse.ArgumentParser()
ap.add_argument("--coco", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--trees", type=int, default=60)
ap.add_argument("--work-scale", type=float, default=0.5, help="predict at this fraction of 3 px/mm (speed)")
a = ap.parse_args()

t0 = time.time()
train = [s for s in iter_samples(a.coco, a.images, splits={"train"})]
usable = [s for s in train if s.cal is not None]
print(f"train photos: {len(train)} (mat found in {len(usable)})")
if len(usable) < 3:
    raise SystemExit("Need at least 3 training photos with the mat visible.")
seg = PixelForestSegmenter(n_estimators=a.trees, work_scale=a.work_scale)
seg.fit((s.rect, s.instances, s.usable) for s in usable)
seg.meta.update({"train_photos": len(usable), "seconds": round(time.time() - t0, 1)})
seg.save(a.out)
print(f"saved {a.out} in {time.time() - t0:.0f}s: {seg.meta}")

models = load_models(a.out, None, None)
val = list(iter_samples(a.coco, a.images, splits={"val"}))
if val:
    r = evaluate(val, models, load_rules())
    write_json(a.out + ".val.json", {k: v for k, v in r.items() if k != "per_photo"})
    print("VAL split (detection is what this model changes):\n" + to_markdown(r))
