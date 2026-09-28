"""Dataset (photos + ground-truth labels), model registry and evaluations."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
import zipfile

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ekmaap import CONDITIONS
from ekmaap.coco import add_annotation, add_image, new_coco
from ekmaap.dataset import make_sample
from ekmaap.evaluate import evaluate
from ekmaap.pipeline import analyse, decode
from ekmaap.rules import RuleSet as Rules

from .. import storage
from ..audit import log
from ..config import settings
from ..db import get_db
from ..models import Annotation, Evaluation, ModelVersion, Photo
from ..security import current_user, require
from ..services.engine import active_rule_set, clear_cache, load_model_file, models_for
from .core import row
from .grading import get_photo, store_photo

router = APIRouter()


# ------------------------------------------------------------------ dataset photos + labels
@router.post("/dataset/photos", status_code=201, tags=["dataset"])
async def upload_dataset_photo(photo: UploadFile = File(...), split_group: str | None = Form(None),
                               db: Session = Depends(get_db), user=Depends(require("labeller", "supervisor"))):
    """Add a photo for training/evaluation (not linked to a lot). split_group = e.g. capture day."""
    p = await store_photo(db, photo, user, "dataset", split_group=split_group)
    log(db, user, "upload", "dataset_photo", p.id); db.commit()
    return row(p, "id", "sha256", "original_name", "width", "height", "split", "split_group")


@router.get("/dataset/photos", tags=["dataset"])
def list_dataset(split: str | None = None, labelled: bool | None = None, db: Session = Depends(get_db), user=Depends(current_user)):
    n_ann = select(Annotation.photo_id, func.count().label("n")).group_by(Annotation.photo_id).subquery()
    st = select(Photo, n_ann.c.n).outerjoin(n_ann, n_ann.c.photo_id == Photo.id).where(Photo.purpose == "dataset").order_by(Photo.created_at)
    if split:
        st = st.where(Photo.split == split)
    out = []
    for p, n in db.execute(st):
        if labelled is not None and bool(n) != labelled:
            continue
        out.append({**row(p, "id", "original_name", "split", "split_group", "width", "height"), "annotations": n or 0})
    return out


class AnnotationIn(BaseModel):
    polygon: list[list[float]] = Field(min_length=3)
    condition: str = Field(pattern="^(sound|sprouted|damaged|rotten)$")
    ruler_mm: float | None = Field(None, gt=0, lt=200)


@router.get("/photos/{photo_id}/annotations", tags=["dataset"])
def get_annotations(photo_id: uuid.UUID, db: Session = Depends(get_db), user=Depends(current_user)):
    return [row(a, "id", "polygon", "condition", "ruler_mm", "source") for a in
            db.scalars(select(Annotation).where(Annotation.photo_id == photo_id).order_by(Annotation.created_at))]


@router.put("/photos/{photo_id}/annotations", tags=["dataset"])
def put_annotations(photo_id: uuid.UUID, items: list[AnnotationIn], source: str = "manual", db: Session = Depends(get_db),
                    user=Depends(require("labeller", "supervisor"))):
    """Replace ALL ground-truth labels of a photo (polygons in original photo pixels)."""
    if source not in ("manual", "prelabel", "import", "review"):
        raise HTTPException(422, "bad source")
    p = get_photo(db, photo_id)
    db.execute(delete(Annotation).where(Annotation.photo_id == p.id))
    for it in items:
        db.add(Annotation(photo_id=p.id, polygon=it.polygon, condition=it.condition, ruler_mm=it.ruler_mm, source=source, labeller_id=user.id))
    log(db, user, "label", "photo", p.id, n=len(items), source=source); db.commit()
    return {"photo_id": str(p.id), "annotations": len(items)}


@router.post("/photos/{photo_id}/prelabel", tags=["dataset"])
def prelabel(photo_id: uuid.UUID, save: bool = False, db: Session = Depends(get_db), user=Depends(require("labeller", "supervisor"))):
    """Suggest outlines + conditions with the active models (faster than drawing). save=true stores them as
    annotations (source=prelabel) - a labeller must still check them."""
    p = get_photo(db, photo_id)
    models, _ = models_for(db)
    res = analyse(decode(storage.read(p.storage_path)), models, Rules(**active_rule_set(db).params), px_per_mm=settings.px_per_mm)
    items = [{"polygon": o["polygon"], "condition": o["condition"], "ruler_mm": None} for o in res["onions"]]
    if save:
        return put_annotations(p.id, [AnnotationIn(**i) for i in items], "prelabel", db, user)
    return {"photo_id": str(p.id), "suggestions": items, "scale_found": res["scale_found"]}


class SplitIn(BaseModel):
    train: float = 0.7
    val: float = 0.15
    reassign: bool = False


@router.post("/dataset/split", tags=["dataset"])
def assign_split(body: SplitIn, db: Session = Depends(get_db), user=Depends(require("supervisor"))):
    """Assign train/val/test by photo (or by split_group). Stable: a photo keeps its split when more are added."""
    st = select(Photo).where(Photo.purpose == "dataset")
    if not body.reassign:
        st = st.where(Photo.split == "unassigned")
    n = {"train": 0, "val": 0, "test": 0}
    for p in db.scalars(st):
        key = p.split_group or str(p.id)
        h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        p.split = "train" if h < body.train else "val" if h < body.train + body.val else "test"
        n[p.split] += 1
    log(db, user, "split", "dataset", None, **n); db.commit()
    return {"assigned": n}


def _labelled_photos(db, split: str | None):
    st = select(Photo).where(Photo.purpose == "dataset", Photo.id.in_(select(Annotation.photo_id)))
    if split:
        st = st.where(Photo.split == split)
    return db.scalars(st.order_by(Photo.created_at)).all()


@router.get("/dataset/export", tags=["dataset"])
def export_dataset(split: str | None = None, db: Session = Depends(get_db), user=Depends(require("labeller", "supervisor"))):
    """ZIP with images/, annotations.json (COCO, categories = conditions, per-image split) and ruler.csv -
    exactly what the scripts/train_*.py and scripts/evaluate.py expect."""
    coco = new_coco()
    buf = io.BytesIO()
    rows = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in _labelled_photos(db, split):
            name = f"{p.id}{storage.path(p.storage_path).suffix}"
            z.writestr(f"images/{name}", storage.read(p.storage_path))
            iid = add_image(coco, name, p.width, p.height, split=p.split, split_group=p.split_group, original_name=p.original_name)
            for a in db.scalars(select(Annotation).where(Annotation.photo_id == p.id)):
                add_annotation(coco, iid, np.asarray(a.polygon), a.condition)
                if a.ruler_mm:
                    c = np.asarray(a.polygon).mean(0)
                    rows.append([name, round(float(c[0]), 1), round(float(c[1]), 1), a.ruler_mm])
        z.writestr("annotations.json", json.dumps(coco))
        s = io.StringIO(); w = csv.writer(s); w.writerow(["file_name", "x", "y", "ruler_mm"]); w.writerows(rows)
        z.writestr("ruler.csv", s.getvalue())
    return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="ekmaap_dataset.zip"'})


@router.post("/dataset/import", tags=["dataset"])
async def import_coco(coco_file: UploadFile = File(...), db: Session = Depends(get_db), user=Depends(require("labeller", "supervisor"))):
    """Import corrected labels from CVAT / Label Studio (COCO). Images are matched by file name to the
    uploaded photo's original name (or to '<photo id>.<ext>' from our own export). Replaces their labels."""
    try:
        coco = json.loads(await coco_file.read())
        cats = {c["id"]: c["name"].strip().lower() for c in coco["categories"]}
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(422, "Not a COCO file")
    bad = set(cats.values()) - set(CONDITIONS)
    if bad:
        raise HTTPException(422, f"Unknown categories {sorted(bad)}; use {list(CONDITIONS)}")
    photos = db.scalars(select(Photo).where(Photo.purpose == "dataset")).all()
    by_name = {}
    for p in photos:
        by_name[p.original_name] = p
        by_name[f"{p.id}{storage.path(p.storage_path).suffix}"] = p
    matched, unmatched = 0, []
    for im in coco["images"]:
        p = by_name.get(im["file_name"]) or by_name.get(im["file_name"].split("/")[-1])
        if not p:
            unmatched.append(im["file_name"]); continue
        sx, sy = p.width / im.get("width", p.width), p.height / im.get("height", p.height)
        # COCO carries no ruler sizes: keep existing ones by re-attaching them to the new outline that contains them
        rulers = [(np.asarray(a.polygon, float).mean(0), a.ruler_mm) for a in
                  db.scalars(select(Annotation).where(Annotation.photo_id == p.id, Annotation.ruler_mm.is_not(None)))]
        db.execute(delete(Annotation).where(Annotation.photo_id == p.id))
        for a in coco["annotations"]:
            if a["image_id"] != im["id"] or not a.get("segmentation") or isinstance(a["segmentation"], dict):
                continue
            poly = max((np.asarray(s_, float).reshape(-1, 2) for s_ in a["segmentation"] if len(s_) >= 6), key=len, default=None)
            if poly is None:
                continue
            poly = poly * [sx, sy]
            ruler = next((mm for c_, mm in rulers if cv2.pointPolygonTest(poly.astype(np.float32), tuple(map(float, c_)), False) >= 0), None)
            db.add(Annotation(photo_id=p.id, polygon=poly.round(1).tolist(), condition=cats[a["category_id"]], ruler_mm=ruler,
                              source="import", labeller_id=user.id))
        matched += 1
    log(db, user, "import", "dataset", None, matched=matched, unmatched=len(unmatched)); db.commit()
    return {"photos_updated": matched, "unmatched_files": unmatched}


# ------------------------------------------------------------------ model registry
BACKENDS = {"segmenter": ("classical", "pixel_forest", "yolo_seg"), "classifier": ("heuristic", "sklearn"), "size_calibration": ("linear",)}
EXT = {"pixel_forest": ".joblib", "sklearn": ".joblib", "yolo_seg": ".pt", "linear": ".json"}


def mv_out(m):
    return row(m, "id", "kind", "backend", "version", "file_sha256", "metrics", "notes", "is_active", "created_at")


@router.get("/models", tags=["models"])
def list_models(db: Session = Depends(get_db), user=Depends(current_user)):
    return [mv_out(m) for m in db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()))]


@router.post("/models", status_code=201, tags=["models"])
async def register_model(kind: str = Form(...), backend: str = Form(...), version: str = Form(...), notes: str | None = Form(None),
                         metrics: str | None = Form(None, description="JSON, e.g. the output of scripts/evaluate.py"),
                         file: UploadFile | None = File(None), db: Session = Depends(get_db), user=Depends(require("supervisor"))):
    """Register a trained model file (from scripts/train_*.py). It is test-loaded before it is accepted."""
    if backend not in BACKENDS.get(kind, ()):
        raise HTTPException(422, f"backend for {kind} must be one of {BACKENDS.get(kind)}")
    m = ModelVersion(kind=kind, backend=backend, version=version, notes=notes, created_by=user.id,
                     metrics=json.loads(metrics) if metrics else None)
    if backend not in ("classical", "heuristic"):
        if file is None:
            raise HTTPException(422, "This backend needs a model file")
        data = await file.read()
        m.file_sha256, m.storage_path = storage.put(data, EXT[backend])
    try:
        load_model_file(m)
    except Exception as e:  # noqa: BLE001 - report any loading problem to the uploader
        raise HTTPException(422, f"Model file could not be loaded: {e}")
    db.add(m)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(409, "Version name already used")
    log(db, user, "register", "model", m.id, kind=kind, version=version); db.commit()
    return mv_out(m)


@router.post("/models/{model_id}/activate", tags=["models"])
def activate_model(model_id: uuid.UUID, db: Session = Depends(get_db), user=Depends(require("supervisor"))):
    m = db.get(ModelVersion, model_id)
    if not m:
        raise HTTPException(404, "Model not found")
    db.execute(update(ModelVersion).where(ModelVersion.kind == m.kind, ModelVersion.is_active).values(is_active=False))
    db.flush()
    m.is_active = True
    log(db, user, "activate", "model", m.id, kind=m.kind, version=m.version); db.commit(); clear_cache()
    return mv_out(m)


# ------------------------------------------------------------------ evaluations
class EvalIn(BaseModel):
    split: str = Field("test", pattern="^(train|val|test)$")


@router.post("/evaluations", status_code=201, tags=["models"])
def run_evaluation(body: EvalIn, db: Session = Depends(get_db), user=Depends(require("supervisor"))):
    """Evaluate the ACTIVE models + rule set on labelled photos of a split (synchronous; keep test sets modest)."""
    rs = active_rule_set(db)
    models, act = models_for(db)
    samples = []
    for p in _labelled_photos(db, body.split):
        anns = db.scalars(select(Annotation).where(Annotation.photo_id == p.id)).all()
        ruler = [(*np.asarray(a.polygon).mean(0).tolist(), a.ruler_mm) for a in anns if a.ruler_mm]
        samples.append(make_sample(str(p.id), str(p.id), p.split, decode(storage.read(p.storage_path)),
                                   [np.asarray(a.polygon) for a in anns], [a.condition for a in anns], ruler, settings.px_per_mm))
    if not samples:
        raise HTTPException(409, f"No labelled photos in split '{body.split}'")
    r = evaluate(samples, models, Rules(**rs.params))
    ev = Evaluation(split=body.split, n_photos=len(samples), metrics=r, rule_set_id=rs.id, created_by=user.id,
                    segmenter_version_id=act["segmenter"].id if "segmenter" in act else None,
                    classifier_version_id=act["classifier"].id if "classifier" in act else None,
                    size_cal_version_id=act["size_calibration"].id if "size_calibration" in act else None)
    db.add(ev); db.flush(); log(db, user, "evaluate", "models", ev.id, split=body.split); db.commit()
    return {"id": str(ev.id), "split": body.split, "n_photos": len(samples), "models": models.versions, "metrics": r}


@router.get("/evaluations", tags=["models"])
def list_evaluations(db: Session = Depends(get_db), user=Depends(current_user)):
    return [row(e, "id", "split", "n_photos", "segmenter_version_id", "classifier_version_id", "size_cal_version_id", "rule_set_id", "metrics", "created_at")
            for e in db.scalars(select(Evaluation).order_by(Evaluation.created_at.desc()))]
