"""End-to-end evaluation on labelled photos: the numbers you are allowed to quote.

Per photo the prediction is matched to the ground truth by mask overlap (IoU >= 0.5, one-to-one).
Reported:
  detection   precision, recall, F1 of onions
  size        mean absolute error vs ruler (where ruler.csv has values) and vs the labelled outline
  condition   confusion matrix + recall/precision per condition, on matched onions
  lot         mean absolute error of Grade A % and URS % per photo (truth graded with the same rule set)
Always evaluate on the TEST split - photos never used for training.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import CONDITIONS
from .coco import rasterize
from .measure import measure_mask
from .pipeline import analyse
from .rules import summarise


def _iou_matrix(gt: list, pred: list, shape, scale=0.5) -> np.ndarray:
    h, w = int(shape[0] * scale), int(shape[1] * scale)
    S = np.diag([scale, scale, 1.0])
    G = rasterize(gt, (h, w), S)
    P = rasterize(pred, (h, w), S)
    iou = np.zeros((len(gt), len(pred)))
    if not len(gt) or not len(pred):
        return iou
    inter = np.zeros((len(gt) + 1, len(pred) + 1))
    np.add.at(inter, (G.ravel(), P.ravel()), 1)
    ga, pa = inter.sum(1), inter.sum(0)
    for i in range(1, len(gt) + 1):
        for j in range(1, len(pred) + 1):
            if inter[i, j]:
                iou[i - 1, j - 1] = inter[i, j] / (ga[i] + pa[j] - inter[i, j])
    return iou


def evaluate(samples, models, rules, iou_t: float = 0.5) -> dict:
    tp = fp = fn = 0
    cm = {a: {b: 0 for b in CONDITIONS} for a in CONDITIONS}
    size_ruler, size_outline, lot_a, lot_urs, rows = [], [], [], [], []
    n_review = 0
    for s in samples:
        a = analyse(s.image, models, rules)                       # s.image is already downscaled
        pred_polys = [np.asarray(o["polygon"]) for o in a["onions"]]
        gt_polys = [p * s.scale for p in s.polygons]
        iou = _iou_matrix(gt_polys, pred_polys, s.image.shape)
        pairs = []
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            pairs = [(i, j) for i, j in zip(r, c) if iou[i, j] >= iou_t]
        tp += len(pairs); fn += len(gt_polys) - len(pairs); fp += len(pred_polys) - len(pairs)
        n_review += sum(1 for o in a["onions"] if o["grade"] == "Review")
        ruler_of = dict(s.ruler)
        gt_d = {}
        for i, j in pairs:
            o = a["onions"][j]
            cm[s.conditions[i]][o["condition"]] += 1
            if s.cal is not None and o["diameter_mm"] is not None:
                if (i + 1) in ruler_of:
                    size_ruler.append(o["diameter_mm"] - ruler_of[i + 1])
                m = measure_mask(s.instances == i + 1, s.ppm)
                if m:
                    gt_d[i] = m.diameter_mm
                    size_outline.append(o["diameter_mm"] - m.diameter_mm)
        # lot-level: grade the truth with the same rules (ruler size if known, else outline size)
        if s.cal is not None:
            truth = []
            for i, cond in enumerate(s.conditions):
                d = ruler_of.get(i + 1)
                if d is None:
                    m = measure_mask(s.instances == i + 1, s.ppm)
                    d = m.diameter_mm if m else None
                truth.append(rules.grade(cond, d, False)[0])
            ts, ps = summarise(truth)["pct"], a["summary"]["pct"]
            lot_a.append(abs(ps["Grade A"] - ts["Grade A"])); lot_urs.append(abs(ps["URS"] - ts["URS"]))
        rows.append({"file": s.file_name, "truth": len(gt_polys), "found": len(pairs), "extra": len(pred_polys) - len(pairs),
                     "scale_found": a["scale_found"], "pct": a["summary"]["pct"]})
    labels = [c for c in CONDITIONS if sum(cm[c].values())]
    per = {}
    for c in labels:
        t = cm[c][c]; n = sum(cm[c].values()); p = sum(cm[x][c] for x in CONDITIONS)
        per[c] = {"n": n, "recall": t / n if n else None, "precision": t / p if p else None}
    mae = lambda v: float(np.mean(np.abs(v))) if v else None
    return {
        "photos": len(rows),
        "detection": {"truth": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                      "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                      "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0},
        "size": {"vs_ruler_mae_mm": mae(size_ruler), "vs_ruler_bias_mm": float(np.mean(size_ruler)) if size_ruler else None, "n_ruler": len(size_ruler),
                 "vs_outline_mae_mm": mae(size_outline), "n_outline": len(size_outline)},
        "condition": {"per_class": per, "accuracy": (sum(cm[c][c] for c in CONDITIONS) / tp) if tp else None,
                      "confusion": {"labels": list(CONDITIONS), "matrix": [[cm[a][b] for b in CONDITIONS] for a in CONDITIONS]}},
        "lot": {"grade_a_pct_mae": mae(lot_a), "urs_pct_mae": mae(lot_urs), "n": len(lot_a)},
        "review": {"onions_sent_to_review": n_review, "share": n_review / max(1, tp + fp)},
        "per_photo": rows,
    }


def to_markdown(r: dict) -> str:
    d, s, c, l = r["detection"], r["size"], r["condition"], r["lot"]
    f = lambda v, fmt="{:.3f}": "-" if v is None else fmt.format(v)
    lines = [f"Photos evaluated: {r['photos']}", "",
             "| Metric | Value |", "| --- | --- |",
             f"| Onions (truth / found / missed / extra) | {d['truth']} / {d['tp']} / {d['fn']} / {d['fp']} |",
             f"| Detection precision / recall / F1 | {f(d['precision'])} / {f(d['recall'])} / {f(d['f1'])} |",
             f"| Size error vs ruler, mean abs (n) | {f(s['vs_ruler_mae_mm'], '{:.2f} mm')} ({s['n_ruler']}) |",
             f"| Size bias vs ruler | {f(s['vs_ruler_bias_mm'], '{:+.2f} mm')} |",
             f"| Size error vs labelled outline (n) | {f(s['vs_outline_mae_mm'], '{:.2f} mm')} ({s['n_outline']}) |",
             f"| Condition accuracy on matched onions | {f(c['accuracy'])} |",
             f"| Grade A % error per photo, mean abs | {f(l['grade_a_pct_mae'], '{:.1f} points')} |",
             f"| URS % error per photo, mean abs | {f(l['urs_pct_mae'], '{:.1f} points')} |",
             f"| Onions sent to officer review | {r['review']['onions_sent_to_review']} ({r['review']['share']:.0%}) |", "",
             "| Condition | n | recall | precision |", "| --- | --- | --- | --- |"]
    for k, v in c["per_class"].items():
        lines.append(f"| {k} | {v['n']} | {f(v['recall'])} | {f(v['precision'])} |")
    return "\n".join(lines)
