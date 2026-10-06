"""Size distribution of a lot: how even are the onions?

Buyers care about uniformity as well as the Grade A share. ``describe`` gives mean, spread, percentiles
and the coefficient of variation (CV = std / mean); ``histogram`` gives counts per size band.
Onions without a size (no reference found, cut off) are ignored and counted separately.
"""
from __future__ import annotations

import numpy as np


def _sizes(analysis_or_list) -> tuple[np.ndarray, int]:
    src = analysis_or_list.get("onions", []) if isinstance(analysis_or_list, dict) else analysis_or_list
    vals = [o.get("diameter_mm") if isinstance(o, dict) else o for o in src]
    good = [float(v) for v in vals if v is not None]
    return np.array(good, float), len(vals) - len(good)


def describe(analysis_or_list) -> dict:
    d, missing = _sizes(analysis_or_list)
    if d.size == 0:
        return {"n": 0, "n_without_size": missing}
    mean = float(d.mean())
    std = float(d.std(ddof=1)) if d.size > 1 else 0.0
    p10, p50, p90 = (float(x) for x in np.percentile(d, [10, 50, 90]))
    return {"n": int(d.size), "n_without_size": missing, "mean_mm": round(mean, 1), "std_mm": round(std, 1),
            "min_mm": round(float(d.min()), 1), "p10_mm": round(p10, 1), "median_mm": round(p50, 1),
            "p90_mm": round(p90, 1), "max_mm": round(float(d.max()), 1),
            "cv_percent": round(100 * std / mean, 1) if mean else None}


def histogram(analysis_or_list, edges=(0, 35, 45, 55, 70, 200)) -> list[dict]:
    """Counts per band [edges[i], edges[i+1]). Default edges follow the demo rule set (35 / 45 mm)."""
    d, _ = _sizes(analysis_or_list)
    counts, _e = np.histogram(d, bins=list(edges))
    return [{"from_mm": edges[i], "to_mm": edges[i + 1], "count": int(c)} for i, c in enumerate(counts)]
