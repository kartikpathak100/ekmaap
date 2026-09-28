"""Shared helpers for the command-line scripts (run them from the repo root: python scripts/<name>.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ekmaap.classifier import load_classifier  # noqa: E402
from ekmaap.measure import SizeCalibration  # noqa: E402
from ekmaap.pipeline import Models  # noqa: E402
from ekmaap.rules import RuleSet  # noqa: E402
from ekmaap.segmentation import load_segmenter  # noqa: E402


def load_models(segmenter=None, classifier=None, size_cal=None) -> Models:
    return Models(load_segmenter(segmenter), load_classifier(classifier),
                  SizeCalibration.load(size_cal) if size_cal else None,
                  {"segmenter_file": segmenter, "classifier_file": classifier, "size_cal_file": size_cal})


def load_rules(path=None) -> RuleSet:
    return RuleSet(**json.loads(Path(path).read_text())) if path else RuleSet()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
