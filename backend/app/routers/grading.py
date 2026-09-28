"""The officer's workflow: lot -> photos -> analysis -> review -> grade -> report; plus disputes and public verify."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ekmaap.pipeline import annotate, decode

from .. import storage
from ..audit import log
from ..config import settings
from ..db import get_db
from ..models import Analysis, Detection, DetectionReview, Dispute, Lot, Photo, Report, User
from ..security import current_user, require
from ..services.engine import latest_analysis, run_analysis
from ..services.reports import build_record, canonical, fingerprint, lot_grading, render_pdf
from .core import row

router = APIRouter()
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


async def store_photo(db: Session, upload: UploadFile, user: User, purpose: str, lot: Lot | None = None,
                      lat: float | None = None, lon: float | None = None, split_group: str | None = None) -> Photo:
    data = await upload.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "Photo too large")
    ext = IMAGE_TYPES.get(upload.content_type or "", None)
    if ext is None:
        raise HTTPException(415, "Upload a JPEG, PNG or WebP photo")
    try:
        img = decode(data)
    except ValueError:
        raise HTTPException(422, "The file is not a readable image")
    sha, rel = storage.put(data, ext)
    p = Photo(lot_id=lot.id if lot else None, purpose=purpose, sha256=sha, storage_path=rel, original_name=upload.filename,
              content_type=upload.content_type, width=img.shape[1], height=img.shape[0], uploaded_by=user.id, lat=lat, lon=lon,
              split_group=split_group)
    db.add(p)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "This exact photo is already in the dataset")
    return p


def analysis_out(a: Analysis) -> dict:
    return {"id": str(a.id), "photo_id": str(a.photo_id), "created_at": a.created_at.isoformat() if a.created_at else None,
            "scale_found": a.scale_found, "calibration_quality": (a.calibration or {}).get("quality"), "summary": a.summary,
            "timing_ms": a.timing_ms,
            "detections": [{"id": str(d.id), "idx": d.idx, "polygon": d.polygon, "centroid": d.centroid, "diameter_mm": d.diameter_mm,
                            "width_mm": d.width_mm, "condition": d.condition, "proba": d.proba, "grade": d.grade, "label": d.label,
                            "needs_review": d.needs_review, "reasons": d.reasons,
                            "review": ({"condition": d.reviews[-1].condition, "diameter_mm": d.reviews[-1].diameter_mm, "note": d.reviews[-1].note}
                                       if d.reviews else None)} for d in a.detections]}


def get_lot(db, lot_id) -> Lot:
    lot = db.get(Lot, lot_id)
    if not lot:
        raise HTTPException(404, "Lot not found")
    return lot


# ------------------------------------------------------------------ lots
class LotIn(BaseModel):
    lot_code: str = Field(min_length=1, max_length=64)
    centre_id: uuid.UUID
    farmer_id: uuid.UUID | None = None
    variety: str | None = None
    declared_weight_kg: float | None = None
    lat: float | None = None
    lon: float | None = None


LOT_FIELDS = ("id", "lot_code", "centre_id", "officer_id", "farmer_id", "variety", "declared_weight_kg", "status", "created_at")


@router.post("/lots", status_code=201, tags=["lots"])
def create_lot(body: LotIn, db: Session = Depends(get_db), user: User = Depends(require("officer", "supervisor"))):
    lot = Lot(officer_id=user.id, **body.model_dump())
    db.add(lot)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(409, "Lot code already exists (or centre/farmer id is wrong)")
    log(db, user, "create", "lot", lot.id); db.commit()
    return row(lot, *LOT_FIELDS)


@router.get("/lots", tags=["lots"])
def list_lots(centre_id: uuid.UUID | None = None, status: str | None = None, limit: int = 50,
              db: Session = Depends(get_db), user: User = Depends(current_user)):
    st = select(Lot).order_by(Lot.created_at.desc()).limit(min(limit, 200))
    if centre_id:
        st = st.where(Lot.centre_id == centre_id)
    if status:
        st = st.where(Lot.status == status)
    return [row(l, *LOT_FIELDS) for l in db.scalars(st)]


@router.get("/lots/{lot_id}", tags=["lots"])
def get_lot_detail(lot_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    lot = get_lot(db, lot_id)
    out = row(lot, *LOT_FIELDS)
    out["photos"] = [{"id": str(p.id), "sha256": p.sha256, "created_at": p.created_at.isoformat()} for p in lot.photos]
    out["reports"] = [{"id": str(r.id), "record_sha256": r.record_sha256, "status": r.status, "issued_at": r.issued_at.isoformat()}
                      for r in db.scalars(select(Report).where(Report.lot_id == lot.id).order_by(Report.issued_at))]
    return out


@router.post("/lots/{lot_id}/photos", status_code=201, tags=["lots"])
async def add_lot_photo(lot_id: uuid.UUID, photo: UploadFile = File(...), lat: float | None = Form(None), lon: float | None = Form(None),
                        db: Session = Depends(get_db), user: User = Depends(require("officer", "supervisor"))):
    """Upload one photo of the lot's sample (on the printed mat). Runs the pipeline and returns the analysis."""
    lot = get_lot(db, lot_id)
    if lot.status in ("reported", "closed"):
        raise HTTPException(409, f"Lot is {lot.status}; open a dispute to re-grade")
    p = await store_photo(db, photo, user, "lot", lot, lat, lon)
    a = run_analysis(db, p)
    log(db, user, "upload+analyse", "photo", p.id, lot=str(lot.id))
    db.commit()
    db.refresh(a)
    return analysis_out(a)


@router.get("/lots/{lot_id}/grade", tags=["lots"])
def preview_grade(lot_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Lot result right now: latest analysis of every photo, officer reviews applied, active rule set."""
    return lot_grading(db, get_lot(db, lot_id))


class IssueIn(BaseModel):
    allow_unreviewed: bool = False


@router.post("/lots/{lot_id}/reports", status_code=201, tags=["reports"])
def issue_report(lot_id: uuid.UUID, body: IssueIn | None = None, db: Session = Depends(get_db),
                 user: User = Depends(require("officer", "supervisor"))):
    """Freeze the lot result into a fingerprinted report. Onions still 'Needs review' block issuing."""
    lot = get_lot(db, lot_id)
    g = lot_grading(db, lot)
    if not g["onions"]:
        raise HTTPException(409, "No graded onions in this lot yet")
    pending = [o for o in g["onions"] if o["grade"] == "Review"]
    if pending and not (body and body.allow_unreviewed and user.role in ("supervisor", "admin")):
        raise HTTPException(409, {"message": f"{len(pending)} onions need officer review first",
                                  "detections": [o["detection_id"] for o in pending]})
    prev = db.scalar(select(Report).where(Report.lot_id == lot.id, Report.status == "valid"))
    rec = build_record(db, lot, user, g, prev)
    text, sha = fingerprint(rec)
    rep = Report(lot_id=lot.id, record=rec, canonical=text, record_sha256=sha, issued_by=user.id, supersedes_id=prev.id if prev else None)
    if prev:
        prev.status = "superseded"
    lot.status = "reported"
    db.add(rep); db.flush()
    log(db, user, "issue", "report", rep.id, sha=sha)
    db.commit()
    return {"id": str(rep.id), "record_sha256": sha, "record": rec}


# ------------------------------------------------------------------ photos / analyses / reviews
def get_photo(db, photo_id) -> Photo:
    p = db.get(Photo, photo_id)
    if not p:
        raise HTTPException(404, "Photo not found")
    return p


@router.get("/photos/{photo_id}", tags=["photos"])
def photo_meta(photo_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = get_photo(db, photo_id)
    return row(p, "id", "lot_id", "purpose", "sha256", "original_name", "width", "height", "split", "split_group", "created_at")


@router.get("/photos/{photo_id}/image", tags=["photos"])
def photo_image(photo_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = get_photo(db, photo_id)
    return Response(storage.read(p.storage_path), media_type=p.content_type or "image/jpeg")


@router.get("/photos/{photo_id}/annotated.jpg", tags=["photos"])
def photo_annotated(photo_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = get_photo(db, photo_id)
    a = latest_analysis(db, p.id)
    if not a:
        raise HTTPException(404, "Photo not analysed yet")
    img = decode(storage.read(p.storage_path))
    view = {"onions": [{"polygon": d.polygon, "centroid": d.centroid, "idx": d.idx, "diameter_mm": d.diameter_mm,
                        "grade": d.grade, "label": d.label} for d in a.detections]}
    ok, jpg = cv2.imencode(".jpg", annotate(img, view), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(jpg.tobytes(), media_type="image/jpeg")


@router.get("/photos/{photo_id}/analysis", tags=["photos"])
def photo_latest_analysis(photo_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    a = latest_analysis(db, get_photo(db, photo_id).id)
    if not a:
        raise HTTPException(404, "Photo not analysed yet")
    return analysis_out(a)


@router.post("/photos/{photo_id}/analyse", status_code=201, tags=["photos"])
def reanalyse(photo_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(require("officer", "supervisor", "labeller"))):
    """Run the CURRENT active models again (new analysis row; old ones are kept)."""
    p = get_photo(db, photo_id)
    if p.lot and p.lot.status in ("reported", "closed"):
        raise HTTPException(409, "Lot already reported")
    a = run_analysis(db, p)
    log(db, user, "reanalyse", "photo", p.id); db.commit(); db.refresh(a)
    return analysis_out(a)


class ReviewIn(BaseModel):
    condition: str = Field(pattern="^(sound|sprouted|damaged|rotten)$")
    diameter_mm: float | None = Field(None, gt=0, lt=200)
    note: str | None = None


@router.post("/detections/{detection_id}/reviews", status_code=201, tags=["photos"])
def review_detection(detection_id: uuid.UUID, body: ReviewIn, db: Session = Depends(get_db),
                     user: User = Depends(require("officer", "supervisor"))):
    """Officer's decision on one onion (confirms or corrects the model). Latest review wins."""
    d = db.get(Detection, detection_id)
    if not d:
        raise HTTPException(404, "Detection not found")
    photo = db.get(Photo, db.get(Analysis, d.analysis_id).photo_id)
    if photo.lot and photo.lot.status in ("reported", "closed"):
        raise HTTPException(409, "Lot already reported; open a dispute to re-grade")
    r = DetectionReview(detection_id=d.id, reviewer_id=user.id, **body.model_dump())
    db.add(r); db.flush()
    log(db, user, "review", "detection", d.id, condition=body.condition, model_condition=d.condition)
    db.commit()
    return row(r, "id", "detection_id", "condition", "diameter_mm", "note", "created_at")


# ------------------------------------------------------------------ reports / disputes / verify
def get_report(db, report_id) -> Report:
    r = db.get(Report, report_id)
    if not r:
        raise HTTPException(404, "Report not found")
    return r


@router.get("/reports/{report_id}", tags=["reports"])
def report_json(report_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    r = get_report(db, report_id)
    return {"id": str(r.id), "status": r.status, "record_sha256": r.record_sha256, "record": json.loads(r.canonical)}


@router.get("/reports/{report_id}/pdf", tags=["reports"])
def report_pdf(report_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    r = get_report(db, report_id)
    return Response(render_pdf(db, r), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="report_{r.record["lot"]["code"]}_{r.record_sha256[:8]}.pdf"'})


class DisputeIn(BaseModel):
    raised_by_name: str
    raised_by_role: str = Field(pattern="^(farmer|officer|trader|other)$")
    reason: str = Field(min_length=3)


@router.post("/reports/{report_id}/disputes", status_code=201, tags=["disputes"])
def open_dispute(report_id: uuid.UUID, body: DisputeIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    r = get_report(db, report_id)
    dsp = Dispute(report_id=r.id, **body.model_dump())
    lot = db.get(Lot, r.lot_id)
    lot.status = "disputed"                    # re-opens photo upload / reviews for a re-grade
    db.add(dsp); db.flush(); log(db, user, "open", "dispute", dsp.id, report=str(r.id)); db.commit()
    return row(dsp, "id", "report_id", "raised_by_name", "raised_by_role", "reason", "status", "created_at")


@router.get("/disputes", tags=["disputes"])
def list_disputes(status: str | None = None, db: Session = Depends(get_db), user: User = Depends(require("officer", "supervisor"))):
    st = select(Dispute).order_by(Dispute.created_at.desc())
    if status:
        st = st.where(Dispute.status == status)
    return [row(d, "id", "report_id", "raised_by_name", "raised_by_role", "reason", "status", "resolution_note", "new_report_id", "created_at", "resolved_at")
            for d in db.scalars(st)]


class DisputeResolve(BaseModel):
    status: str = Field(pattern="^(regraded|upheld|rejected)$")
    resolution_note: str
    new_report_id: uuid.UUID | None = None


@router.patch("/disputes/{dispute_id}", tags=["disputes"])
def resolve_dispute(dispute_id: uuid.UUID, body: DisputeResolve, db: Session = Depends(get_db), user: User = Depends(require("supervisor"))):
    d = db.get(Dispute, dispute_id)
    if not d:
        raise HTTPException(404, "Dispute not found")
    if body.status == "regraded" and not body.new_report_id:
        raise HTTPException(422, "A re-grade must point to the new report")
    d.status, d.resolution_note, d.new_report_id = body.status, body.resolution_note, body.new_report_id
    d.resolved_by, d.resolved_at = user.id, datetime.now(timezone.utc)
    rep = db.get(Report, d.report_id)
    lot = db.get(Lot, rep.lot_id)
    if lot.status == "disputed":
        lot.status = "reported"
    log(db, user, "resolve", "dispute", d.id, status=body.status); db.commit()
    return row(d, "id", "status", "resolution_note", "new_report_id", "resolved_at")


@router.get("/verify/{record_sha256}", tags=["verify (public)"])
def verify_lookup(record_sha256: str, db: Session = Depends(get_db)):
    """Public: is there an issued report with this fingerprint? (What the QR code on the PDF encodes.)"""
    r = db.scalar(select(Report).where(Report.record_sha256 == record_sha256.lower()))
    if not r:
        return {"found": False}
    return {"found": True, "status": r.status, "lot": r.record["lot"]["code"], "issued_at": r.record["issued_at"],
            "summary": r.record["summary"], "rule_set": r.record["rule_set"]["version"]}


@router.post("/verify", tags=["verify (public)"])
async def verify_record(record: UploadFile = File(..., description="the report record JSON"),
                        photos: list[UploadFile] = File(default=[]), db: Session = Depends(get_db)):
    """Public: recompute the fingerprint of a record (and optionally of photos) and compare with what was issued."""
    try:
        rec = json.loads(await record.read())
    except json.JSONDecodeError:
        raise HTTPException(422, "record is not JSON")
    rec = rec.get("record", rec)                       # accept the GET /reports/{id} response too
    _, sha = fingerprint(rec)
    stored = db.scalar(select(Report).where(Report.record_sha256 == sha))
    import hashlib
    photo_checks = []
    wanted = {p["sha256"] for p in rec.get("photos", [])}
    for f in photos:
        h = hashlib.sha256(await f.read()).hexdigest()
        photo_checks.append({"file": f.filename, "sha256": h, "in_record": h in wanted})
    return {"fingerprint": sha, "issued": stored is not None, "status": stored.status if stored else None,
            "unchanged": stored is not None and stored.canonical == canonical(rec), "photos": photo_checks}
