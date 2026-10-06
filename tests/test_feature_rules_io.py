import json
import pytest
from ekmaap.rules import RuleSet
from ekmaap.rules_io import diff, load, save


def test_roundtrip(tmp_path):
    p = tmp_path / "r.json"
    r = RuleSet(grade_a_min_mm=50, urs_min_mm=40, version="t1")
    save(r, p)
    assert load(p) == r


def test_diff_lists_changes_only():
    d = diff(RuleSet(), RuleSet(grade_a_min_mm=50, version="v2"))
    assert set(x.split(":")[0] for x in d) == {"grade_a_min_mm", "version"}
    assert diff(RuleSet(), RuleSet()) == []


def test_invalid_values_give_readable_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"grade_a_min_mm": 30, "urs_min_mm": 40}))
    with pytest.raises(ValueError, match="invalid rule set"):
        load(p)
    p.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load(p)
