"""Save, load and compare grading rule sets as JSON files.

The official grade limits are not public yet. The department may send them as a document;
an officer or developer then writes them into a JSON file, loads it here (the values are validated by
``RuleSet``), and ``diff`` shows exactly what changed from the previous version.
"""
from __future__ import annotations

import json
from pathlib import Path

from .rules import RuleSet


def save(rs: RuleSet, path: str | Path) -> None:
    Path(path).write_text(json.dumps(rs.model_dump(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load(path: str | Path) -> RuleSet:
    """Load and validate. Raises ValueError with a readable message on bad input."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    try:
        return RuleSet(**data)
    except Exception as e:                      # pydantic ValidationError -> plain message
        raise ValueError(f"{path}: invalid rule set: {e}") from e


def diff(old: RuleSet, new: RuleSet) -> list[str]:
    """Human-readable list of changed fields, e.g. 'grade_a_min_mm: 45 -> 50'."""
    a, b = old.model_dump(), new.model_dump()
    return [f"{k}: {a[k]} -> {b[k]}" for k in b if a.get(k) != b[k]]
