import cv2
import numpy as np
from ekmaap import synth
from ekmaap.quality_checks import QualityLimits, check


def test_good_synthetic_photo_passes():
    img, _ = synth.make_photo(seed=1)
    r = check(img)
    assert r.ok, r.warnings


def test_blur_is_flagged():
    img, _ = synth.make_photo(seed=1)
    blurred = cv2.GaussianBlur(img, (0, 0), 12)
    r = check(blurred)
    assert not r.ok and any("blurred" in w for w in r.warnings)


def test_dark_and_bright_flagged():
    img, _ = synth.make_photo(seed=2)
    assert any("too dark" in w for w in check((img * 0.2).astype(np.uint8)).warnings)
    assert any("too bright" in w for w in check(np.clip(img.astype(int) + 120, 0, 255).astype(np.uint8)).warnings)


def test_glare_and_small():
    img = np.full((600, 700, 3), 255, np.uint8)
    r = check(img, QualityLimits())
    msgs = " ".join(r.warnings)
    assert "glare" in msgs and "small" in msgs
