"""Train the condition classifier (sound / sprouted / damaged / rotten) on labelled onions.

    python scripts/train_classifier.py --coco data/real/annotations.json --images data/real/images \
        --segmenter models/segmenter_pf.joblib --out models/classifier.joblib

Training examples come from the train + val splits:
  * features inside each LABELLED outline, and
  * (with --segmenter) features inside the SEGMENTER's outline of the same onion (matched by overlap),
    because in the field the classifier only ever sees the segmenter's outlines.
Cross-validation holds out whole photos; the test split is left for scripts/evaluate.py.
"""
import argparse

import numpy as np

from _common import load_models, write_json
from ekmaap.classifier import LearnedClassifier
from ekmaap.dataset import iter_samples
from ekmaap.features import onion_features, vector
from ekmaap.measure import measure_mask

ap = argparse.ArgumentParser()
ap.add_argument("--coco", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--segmenter", help="also learn from this segmenter's outlines (recommended)")
a = ap.parse_args()

seg = load_models(a.segmenter).segmenter if a.segmenter else None


def feats(rect, m, ppm):
    meas = measure_mask(m, ppm)
    if meas is None:
        return None
    return vector(onion_features(rect, m, meas.solidity, meas.width_mm / max(meas.diameter_mm, 1e-6)))


X, y, g = [], [], []
skipped = n_pred = 0
for s in iter_samples(a.coco, a.images, splits={"train", "val"}):
    if s.cal is None:
        skipped += 1
        continue
    for k, cond in enumerate(s.conditions, 1):
        v = feats(s.rect, s.instances == k, s.ppm)
        if v is not None:
            X.append(v); y.append(cond); g.append(s.file_name)
    if seg is not None:
        pred = seg.segment(s.rect, s.ppm, s.usable, None)
        for j in range(1, int(pred.max()) + 1):
            pm = pred == j
            gt_ids, counts = np.unique(s.instances[pm], return_counts=True)
            best = max(((gi, c) for gi, c in zip(gt_ids, counts) if gi > 0), key=lambda t: t[1], default=None)
            if best is None:
                continue
            gi, inter = best
            iou = inter / (pm.sum() + (s.instances == gi).sum() - inter)
            if iou >= 0.5:
                v = feats(s.rect, pm, s.ppm)
                if v is not None:
                    X.append(v); y.append(s.conditions[gi - 1]); g.append(s.file_name); n_pred += 1
if not X:
    raise SystemExit("No usable labelled onions (is the mat visible in the photos?)")
clf = LearnedClassifier.train(np.array(X), np.array(y), np.array(g))
clf.meta["examples_from_segmenter_outlines"] = n_pred
clf.save(a.out)
write_json(a.out + ".metrics.json", clf.to_json())
print(f"saved {a.out}; examples per condition {clf.meta['counts']} ({n_pred} from segmenter outlines); photos skipped (no mat): {skipped}")
cv = clf.meta["cv"]
if cv:
    print(f"cross-validated ({cv['scheme']}): accuracy {cv['accuracy']:.3f}")
    for c, v in cv["per_class"].items():
        print(f"  {c:9s} n={v['n']:4d} recall={v['recall']:.3f} precision={v['precision']:.3f}")
else:
    print("fewer than 3 photos: no cross-validation possible - label more photos")
