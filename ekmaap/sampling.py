"""Lot-level estimate from several photos, with a 95 % interval.

One photo shows only a few dozen onions. A lot is judged from several sample photos, so the
report should say how sure the lot-level percentage is. ``combine`` pools the per-photo counts
and gives a Wilson score interval for each grade share.
"""
from __future__ import annotations

from math import sqrt

GRADES = ("Grade A", "URS", "Reject", "Review")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (proportions in 0..1)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def combine(summaries: list[dict[str, int]]) -> dict:
    """Pool per-photo grade counts, e.g. [{"Grade A": 5, "URS": 3}, {"Grade A": 7, "Reject": 1}].

    Returns total onions, per-grade count, share and 95 % interval (as percentages), and the
    number of photos. Onions flagged 'Review' stay in the total: they are not silently dropped.
    """
    counts = {g: 0 for g in GRADES}
    for s in summaries:
        for g, c in s.items():
            if c < 0:
                raise ValueError("counts must not be negative")
            counts[g] = counts.get(g, 0) + int(c)
    n = sum(counts.values())
    out = {}
    for g, c in counts.items():
        lo, hi = wilson(c, n)
        out[g] = {"count": c, "percent": round(100 * c / n, 1) if n else 0.0,
                  "ci95_low": round(100 * lo, 1), "ci95_high": round(100 * hi, 1)}
    return {"photos": len(summaries), "onions": n, "grades": out}


def photos_needed(margin_pct: float = 5.0, expected_pct: float = 50.0, onions_per_photo: int = 30) -> int:
    """Photos needed so a share near ``expected_pct`` has a 95 % half-width of about ``margin_pct``.

    Simple-random-sampling formula n = z^2 p(1-p) / e^2. Onions in one photo are not fully
    independent (same light, same pile), so treat the result as a minimum.
    """
    if not (0 < margin_pct < 100 and 0 < expected_pct < 100 and onions_per_photo > 0):
        raise ValueError("invalid arguments")
    p, e = expected_pct / 100, margin_pct / 100
    n = (1.96 ** 2) * p * (1 - p) / (e * e)
    return max(1, int(-(-n // onions_per_photo)))
