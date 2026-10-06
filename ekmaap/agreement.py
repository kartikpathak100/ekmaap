"""Grader-agreement study: how often does EkMaap agree with human graders?

The field-validation plan compares the system's labels with officers' labels on the same onions.
``compare`` gives raw agreement, Cohen's kappa (agreement beyond chance) and a confusion table.
Kappa guide (Landis & Koch): <0 poor, 0-0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 almost perfect.
Needs real paired labels. Do not quote it from synthetic data as evidence of field accuracy.
"""
from __future__ import annotations

from collections import Counter


def kappa_band(k: float) -> str:
    for hi, name in ((0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k <= hi:
            return name
    return "almost perfect"


def compare(a: list[str], b: list[str]) -> dict:
    """a, b: labels for the same onions in the same order (e.g. officer vs system)."""
    if len(a) != len(b):
        raise ValueError("label lists must have the same length")
    n = len(a)
    if n == 0:
        raise ValueError("no labels to compare")
    agree = sum(x == y for x, y in zip(a, b))
    po = agree / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    labels = sorted(set(a) | set(b))
    table = {x: {y: 0 for y in labels} for x in labels}
    for x, y in zip(a, b):
        table[x][y] += 1
    return {"n": n, "agreement_percent": round(100 * po, 1), "kappa": round(kappa, 3),
            "kappa_band": kappa_band(kappa), "confusion": table}
