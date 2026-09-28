"""Loads the active models and rule set, runs the pipeline, stores analyses."""
from __future__ import annotations

import threading

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekmaap.classifier import HeuristicClassifier, LearnedClassifier
from ekmaap.measure import SizeCalibration
from ekmaap.pipeline import Models, analyse, decode
from ekmaap.rules import RuleSet as Rules
from ekmaap.segmentation import ClassicalSegmenter, load_segmenter

from .. import storage
from ..config import settings
from ..models import Analysis, Detection, ModelVersion, Photo, RuleSet

_cache: dict = {}
_lock = threading.Lock()


def active_rule_set(db: Session) -> RuleSet:
    rs = db.scalar(select(RuleSet).where(RuleSet.is_active))
    if rs is None:
        raise HTTPException(409, "No active rule set. POST /api/v1/rulesets and activate one.")
    return rs


def active_versions(db: Session) -> dict:
    rows = db.scalars(select(ModelVersion).where(ModelVersion.is_active)).all()
    return {r.kind: r for r in rows}


def load_model_file(mv: ModelVersion):
    """Load one registered model version (also used to validate uploads)."""
    p = storage.path(mv.storage_path) if mv.storage_path else None
    if mv.kind == "segmenter":
        return ClassicalSegmenter() if mv.backend == "classical" else load_segmenter(p)
    if mv.kind == "classifier":
        return HeuristicClassifier() if mv.backend == "heuristic" else LearnedClassifier.load(p)
    if mv.kind == "size_calibration":
        return SizeCalibration.load(p)
    raise ValueError(mv.kind)


def models_for(db: Session) -> tuple[Models, dict]:
    act = active_versions(db)
    key = tuple(str(act[k].id) if k in act else None for k in ("segmenter", "classifier", "size_calibration"))
    with _lock:
        if key not in _cache:
            _cache.clear()
            m = Models()
            try:
                if "segmenter" in act:
                    m.segmenter = load_model_file(act["segmenter"])
                if "classifier" in act:
                    m.classifier = load_model_file(act["classifier"])
                if "size_calibration" in act:
                    m.size_cal = load_model_file(act["size_calibration"])
            except Exception as e:  # noqa: BLE001
                raise HTTPException(503, f"An active model cannot be loaded ({e}). Activate another version.")
            m.versions = {k: act[k].version for k in act}
            _cache[key] = m
        return _cache[key], act


def clear_cache():
    with _lock:
        _cache.clear()


def run_analysis(db: Session, photo: Photo) -> Analysis:
    rs = active_rule_set(db)
    models, act = models_for(db)
    img = decode(storage.read(photo.storage_path))
    res = analyse(img, models, Rules(**rs.params), px_per_mm=settings.px_per_mm)
    a = Analysis(photo_id=photo.id, rule_set_id=rs.id, scale_found=res["scale_found"], calibration=res["calibration"],
                 summary=res["summary"], timing_ms=res["timing_ms"],
                 segmenter_version_id=act["segmenter"].id if "segmenter" in act else None,
                 classifier_version_id=act["classifier"].id if "classifier" in act else None,
                 size_cal_version_id=act["size_calibration"].id if "size_calibration" in act else None)
    for o in res["onions"]:
        a.detections.append(Detection(idx=o["idx"], polygon=o["polygon"], centroid=o["centroid"], diameter_mm=o["diameter_mm"],
                                      width_mm=o["width_mm"], condition=o["condition"], proba=o["proba"], grade=o["grade"],
                                      label=o["label"], needs_review=o["review"], reasons=o["reasons"], features=o["features"]))
    db.add(a)
    db.flush()
    return a


def latest_analysis(db: Session, photo_id) -> Analysis | None:
    return db.scalar(select(Analysis).where(Analysis.photo_id == photo_id).order_by(Analysis.created_at.desc()).limit(1))
