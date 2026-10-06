import pytest
from ekmaap.agreement import compare, kappa_band


def test_perfect_and_none():
    r = compare(["A", "B", "A", "B"], ["A", "B", "A", "B"])
    assert r["agreement_percent"] == 100.0 and r["kappa"] == 1.0
    r = compare(["A", "A", "B", "B"], ["B", "B", "A", "A"])
    assert r["kappa"] == -1.0 and r["kappa_band"] == "poor"


def test_known_kappa():
    # 20 onions: both say A 10, both say B 5, officer A / system B 3, officer B / system A 2 -> po 0.75
    a = ["A"] * 10 + ["B"] * 5 + ["A"] * 3 + ["B"] * 2
    b = ["A"] * 10 + ["B"] * 5 + ["B"] * 3 + ["A"] * 2
    r = compare(a, b)
    assert r["agreement_percent"] == 75.0
    assert abs(r["kappa"] - 0.468) < 0.002 and r["kappa_band"] == "moderate"
    assert r["confusion"]["A"]["B"] == 3 and r["confusion"]["B"]["A"] == 2


def test_errors_and_bands():
    with pytest.raises(ValueError):
        compare(["A"], ["A", "B"])
    with pytest.raises(ValueError):
        compare([], [])
    assert kappa_band(0.9) == "almost perfect"
