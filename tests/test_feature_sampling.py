import pytest
from ekmaap.sampling import combine, photos_needed, wilson


def test_wilson_known_value():
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.404) < 0.002 and abs(hi - 0.596) < 0.002


def test_wilson_edges():
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2


def test_combine_pools_photos():
    r = combine([{"Grade A": 6, "URS": 3, "Reject": 1}, {"Grade A": 4, "URS": 5, "Review": 1}])
    assert r["photos"] == 2 and r["onions"] == 20
    assert r["grades"]["Grade A"]["count"] == 10 and r["grades"]["Grade A"]["percent"] == 50.0
    g = r["grades"]["Grade A"]
    assert g["ci95_low"] < 50.0 < g["ci95_high"]


def test_combine_rejects_negative():
    with pytest.raises(ValueError):
        combine([{"Grade A": -1}])


def test_photos_needed():
    assert photos_needed(5, 50, 30) == 13      # 384.2 onions / 30 per photo, rounded up
    assert photos_needed(10, 50, 30) == 4
