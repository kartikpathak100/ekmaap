"""Export an analysis (output of ``ekmaap.pipeline.analyse``) to CSV and a plain-text summary.

Officers and auditors often want the numbers in a spreadsheet. The CSV has one row per onion;
the summary is a short text block that can be pasted into a message or printed.
"""
from __future__ import annotations

import csv
import io

COLUMNS = ("idx", "diameter_mm", "width_mm", "condition", "grade", "label", "review", "reasons")


def onions_to_csv(analysis: dict) -> str:
    """One row per onion. Missing sizes are written as an empty cell, never as 0."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for o in analysis.get("onions", []):
        row = []
        for c in COLUMNS:
            v = o.get(c)
            if c == "reasons":
                v = "; ".join(v or [])
            elif c == "review":
                v = "yes" if v else "no"
            row.append("" if v is None else v)
        w.writerow(row)
    return buf.getvalue()


def summary_text(analysis: dict) -> str:
    """Short text summary: counts, grade percentages, rule set and whether a size reference was found."""
    onions = analysis.get("onions", [])
    n = len(onions)
    rs = analysis.get("rule_set", {})
    lines = [f"Onions found: {n}",
             f"Size reference found: {'yes' if analysis.get('scale_found') else 'NO - sizes unknown'}"]
    counts: dict[str, int] = {}
    for o in onions:
        counts[o["grade"]] = counts.get(o["grade"], 0) + 1
    for g in ("Grade A", "URS", "Reject", "Review"):
        c = counts.get(g, 0)
        pct = f"{100 * c / n:.1f}%" if n else "-"
        lines.append(f"{g}: {c} ({pct})")
    lines.append(f"Rule set: {rs.get('name', '?')} [{rs.get('version', '?')}]")
    n_rev = sum(1 for o in onions if o.get("review"))
    if n_rev:
        lines.append(f"{n_rev} onion(s) need an officer's decision before a report can be issued.")
    return "\n".join(lines)
