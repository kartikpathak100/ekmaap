from ekmaap.size_stats import describe, histogram


def test_describe_basic_and_missing():
    r = describe([40, 50, 60, None])
    assert r["n"] == 3 and r["n_without_size"] == 1 and r["mean_mm"] == 50.0
    assert r["std_mm"] == 10.0 and r["cv_percent"] == 20.0 and r["median_mm"] == 50.0


def test_describe_empty_and_single():
    assert describe([])["n"] == 0
    assert describe([None])["n_without_size"] == 1
    assert describe([48])["std_mm"] == 0.0


def test_accepts_analysis_dict():
    a = {"onions": [{"diameter_mm": 44.9}, {"diameter_mm": 45.0}, {"diameter_mm": None}]}
    h = histogram(a)
    by = {(x["from_mm"], x["to_mm"]): x["count"] for x in h}
    assert by[(35, 45)] == 1 and by[(45, 55)] == 1 and sum(by.values()) == 2
