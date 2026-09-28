"""ML package tests on synthetic photos (they prove the code works, not real-onion accuracy)."""
import numpy as np
import pytest

from ekmaap import calibration as C, synth
from ekmaap.classifier import HeuristicClassifier, LearnedClassifier
from ekmaap.coco import assign_splits, mask_to_polygon, new_coco, add_image, rasterize
from ekmaap.measure import SizeCalibration, measure_mask
from ekmaap.pipeline import Models, analyse
from ekmaap.rules import RuleSet, summarise


def test_calibration_recovers_scale_under_tilt():
    for seed in range(3):
        img, onions = synth.make_photo(seed=seed, tilt=0.15)
        cal = C.calibrate(img)
        assert cal is not None and cal.mode == "mat" and len(cal.markers) == 4
        assert cal.reproj_mm < 0.5 and cal.quality == "good"


def test_no_mat_returns_none():
    img = np.full((600, 800, 3), 200, np.uint8)
    assert C.calibrate(img) is None


def test_measure_circle_exact():
    ppm = 3.0
    yy, xx = np.mgrid[0:400, 0:400]
    m = (xx - 200) ** 2 + (yy - 200) ** 2 <= (30 * ppm) ** 2          # 60 mm disc
    meas = measure_mask(m, ppm)
    assert abs(meas.diameter_mm - 60) < 0.8


def test_rules_grading():
    r = RuleSet()
    assert r.grade("sound", 50, False) == ("Grade A", "Grade A")
    assert r.grade("sound", 40, False) == ("URS", "Undersized")
    assert r.grade("sound", 30, False) == ("Reject", "Undersized")
    assert r.grade("damaged", 50, False) == ("URS", "Damaged")
    assert r.grade("damaged", 30, False)[0] == "Reject"            # worse of condition and size
    assert r.grade("rotten", 60, False)[0] == "Reject"
    assert r.grade("sound", 60, True) == ("Review", "Needs review")
    assert r.grade("sound", None, False)[0] == "Review"           # no scale -> officer decides
    with pytest.raises(ValueError):
        RuleSet(grade_a_min_mm=40, urs_min_mm=45)
    s = summarise(["Grade A", "Grade A", "URS", "Review"])
    assert s["pct"]["Grade A"] == 50.0 and s["counts"]["Review"] == 1


def test_pipeline_finds_onions_and_sizes():
    img, onions = synth.make_photo(seed=3, touching_pairs=0)
    a = analyse(img, Models(), RuleSet())
    assert a["scale_found"]
    assert len(a["onions"]) == len(onions)
    for o in onions:
        c = o.polygon_img.mean(0)
        best = min(a["onions"], key=lambda q: np.hypot(*(np.array(q["centroid"]) - c)))
        assert abs(best["diameter_mm"] - o.d_mm) < 2.5


def test_size_calibration_robust_fit():
    rng = np.random.default_rng(0)
    true = rng.uniform(30, 70, 60)
    meas = true * 1.05 + 1.0 + rng.normal(0, 0.3, 60)
    meas[:3] += 40                                                  # three merged-onion outliers
    cal = SizeCalibration.fit(meas, true, np.repeat(np.arange(12), 5))
    assert cal.n_outliers == 3 and cal.helps
    assert abs(cal.apply(1.05 * 50 + 1.0) - 50) < 0.3


def test_classifier_train_and_grouped_cv():
    rng = np.random.default_rng(0)
    from ekmaap.features import NAMES
    X = rng.normal(size=(80, len(NAMES)))
    y = np.array(["sound", "rotten"] * 40)
    X[y == "rotten", NAMES.index("dark_frac")] += 5
    clf = LearnedClassifier.train(X, y, np.repeat(np.arange(8), 10))
    assert clf.meta["cv"]["accuracy"] > 0.9
    f = dict(zip(NAMES, X[1])); p = clf.predict(f)
    assert p.condition == "rotten"
    h = HeuristicClassifier().predict({"dark_frac": 0.3, "green_frac": 0, "pale_frac": 0, "solidity": 0.99})
    assert h.condition == "rotten"


def test_splits_by_photo_are_stable():
    coco = new_coco()
    for i in range(50):
        add_image(coco, f"p{i}.jpg", 10, 10)
    a = assign_splits(coco)
    add_image(coco, "new.jpg", 10, 10)
    b = assign_splits(coco)
    assert all(a[k] == b[k] for k in a)
    assert set(a.values()) == {"train", "val", "test"}


def test_polygon_roundtrip():
    lab = rasterize([np.array([[10, 10], [60, 10], [60, 40], [10, 40]], float)], (80, 80))
    poly = mask_to_polygon(lab == 1)
    assert poly is not None and len(poly) >= 4
