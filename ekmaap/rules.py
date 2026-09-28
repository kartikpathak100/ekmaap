"""Grading rule set: size + condition -> Grade A / URS / Reject.

Rules live outside the models. When procurement norms change, a new rule-set
version is created; models are not retrained. Every analysis and report records
the rule-set version it used.

DEFAULT VALUES ARE DEMO VALUES from 2026 news reports (45 mm Grade A minimum after
NAFED's June 2026 relaxation; 35 mm from the 35-70 mm range reported by ETV Bharat).
They are NOT the official specification - replace them once DoCA / NAFED confirm it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Grade = Literal["Grade A", "URS", "Reject"]
RANK = {"Grade A": 0, "URS": 1, "Reject": 2}


class RuleSet(BaseModel):
    name: str = "Demo rule set - 2026 news values, NOT official"
    version: str = "demo-2026-09-28"
    grade_a_min_mm: float = Field(45, gt=0)
    urs_min_mm: float = Field(35, gt=0)
    grade_a_max_mm: float | None = None           # set if an upper limit is ever specified
    map_sprouted: Grade = "Reject"
    map_damaged: Grade = "URS"
    map_rotten: Grade = "Reject"
    review_below_confidence: float = Field(0.6, ge=0, le=1)
    basis: Literal["count"] = "count"            # weight needs a scale - not supported yet

    @model_validator(mode="after")
    def _check(self):
        if self.urs_min_mm > self.grade_a_min_mm:
            raise ValueError("urs_min_mm must not exceed grade_a_min_mm")
        return self

    def size_grade(self, d_mm: float) -> Grade:
        if self.grade_a_max_mm is not None and d_mm > self.grade_a_max_mm:
            return "URS"
        return "Grade A" if d_mm >= self.grade_a_min_mm else "URS" if d_mm >= self.urs_min_mm else "Reject"

    def grade(self, condition: str, diameter_mm: float | None, review: bool) -> tuple[str, str]:
        """Returns (grade, label). grade may be 'Review'; label is one of Grade A / Undersized / Sprouted /
        Damaged / Rotten / Needs review."""
        if review:
            return "Review", "Needs review"
        if condition != "sound":
            g = getattr(self, f"map_{condition}")
            if diameter_mm is not None:
                s = self.size_grade(diameter_mm)
                g = g if RANK[g] >= RANK[s] else s
            return g, condition.capitalize()
        if diameter_mm is None:
            return "Review", "Needs review"
        g = self.size_grade(diameter_mm)
        return g, "Grade A" if g == "Grade A" else "Undersized"


def summarise(grades: list[str]) -> dict:
    keys = ["Grade A", "URS", "Reject", "Review"]
    n = len(grades)
    counts = {k: sum(1 for g in grades if g == k) for k in keys}
    return {"count": n, "counts": counts, "pct": {k: round(100 * counts[k] / n, 1) if n else 0.0 for k in keys}, "basis": "count"}
