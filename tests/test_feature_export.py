import csv, io
from ekmaap.export import COLUMNS, onions_to_csv, summary_text

A = {"scale_found": True, "rule_set": {"name": "R", "version": "v1"},
     "onions": [
         {"idx": 1, "diameter_mm": 52.3, "width_mm": 48.0, "condition": "sound", "grade": "Grade A", "label": "Grade A", "review": False, "reasons": []},
         {"idx": 2, "diameter_mm": None, "width_mm": None, "condition": "sound", "grade": "Review", "label": "Needs review", "review": True, "reasons": ["cut off by the photo edge", "x"]},
     ]}


def test_csv_rows_and_blank_size():
    rows = list(csv.reader(io.StringIO(onions_to_csv(A))))
    assert tuple(rows[0]) == COLUMNS and len(rows) == 3
    assert rows[1][1] == "52.3" and rows[2][1] == ""          # missing size is blank, not 0
    assert rows[2][6] == "yes" and rows[2][7] == "cut off by the photo edge; x"


def test_summary_text():
    t = summary_text(A)
    assert "Onions found: 2" in t and "Grade A: 1 (50.0%)" in t and "1 onion(s) need" in t


def test_summary_without_scale():
    t = summary_text({"scale_found": False, "onions": []})
    assert "NO - sizes unknown" in t and "Grade A: 0 (-)" in t
