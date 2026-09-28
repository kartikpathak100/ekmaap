"""Lot grading (with officer reviews), report records, fingerprints and PDF."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone

import cv2
import numpy as np
from sqlalchemy.orm import Session

from ekmaap.pipeline import annotate, decode
from ekmaap.rules import RuleSet as Rules, summarise

from .. import storage
from ..models import Centre, Farmer, Lot, Report, User
from .engine import active_rule_set, latest_analysis


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(obj) -> tuple[str, str]:
    text = canonical(obj)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def lot_grading(db: Session, lot: Lot) -> dict:
    """Every onion in the lot's latest analyses, officer reviews applied, graded with the ACTIVE rule set."""
    rs = active_rule_set(db)
    rules = Rules(**rs.params)
    photos, onions = [], []
    for p in lot.photos:
        a = latest_analysis(db, p.id)
        if a is None:
            continue
        photos.append({"photo_id": str(p.id), "sha256": p.sha256, "analysis_id": str(a.id), "scale_found": a.scale_found,
                       "calibration_quality": (a.calibration or {}).get("quality")})
        for d in a.detections:
            rev = d.reviews[-1] if d.reviews else None
            cond = rev.condition if rev else d.condition
            dia = rev.diameter_mm if (rev and rev.diameter_mm) else d.diameter_mm
            grade, label = rules.grade(cond, dia, False if rev else d.needs_review)
            onions.append({"photo_id": str(p.id), "detection_id": str(d.id), "idx": d.idx, "diameter_mm": dia,
                           "condition": cond, "grade": grade, "label": label, "officer_reviewed": rev is not None,
                           "model_condition": d.condition})
    return {"rule_set": {"id": str(rs.id), "version": rs.version, "params": rs.params}, "photos": photos, "onions": onions,
            "summary": summarise([o["grade"] for o in onions])}


def build_record(db: Session, lot: Lot, user: User, grading: dict, supersedes: Report | None) -> dict:
    centre = db.get(Centre, lot.centre_id)
    farmer = db.get(Farmer, lot.farmer_id) if lot.farmer_id else None
    officer = db.get(User, lot.officer_id)
    return {
        "schema": "ekmaap.report/1",
        "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "issued_by": user.name,
        "lot": {"code": lot.lot_code, "variety": lot.variety, "centre": centre.code if centre else None,
                "centre_name": centre.name if centre else None, "officer": officer.name if officer else None,
                "farmer": farmer.name if farmer else None},
        "photos": grading["photos"],
        "rule_set": {"version": grading["rule_set"]["version"], "params": grading["rule_set"]["params"]},
        "onions": [{k: o[k] for k in ("photo_id", "idx", "diameter_mm", "condition", "grade", "label", "officer_reviewed")} for o in grading["onions"]],
        "summary": grading["summary"],
        "supersedes": supersedes.record_sha256 if supersedes else None,
        "notice": "Percentages by count of onions photographed. Grade limits come from the rule set version shown.",
    }


def render_pdf(db: Session, report: Report) -> bytes:
    from reportlab.graphics import renderPDF
    from reportlab.graphics.barcode.qr import QrCodeWidget
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    r = report.record
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 20 * mm
    c.setFont("Helvetica-Bold", 16); c.drawString(20 * mm, y, "Onion quality report"); y -= 8 * mm
    c.setFont("Helvetica", 10)
    L = r["lot"]
    for line in (f"Lot {L['code']}  |  Centre {L.get('centre_name') or L.get('centre') or '-'}  |  Officer {L.get('officer') or '-'}",
                 f"Farmer {L.get('farmer') or '-'}  |  Variety {L.get('variety') or '-'}  |  Issued {r['issued_at']} by {r['issued_by']}",
                 f"Rule set {r['rule_set']['version']}  |  status: {report.status}" + (f"  |  supersedes {r['supersedes'][:16]}..." if r.get("supersedes") else "")):
        c.drawString(20 * mm, y, line); y -= 5.5 * mm
    y -= 3 * mm
    s = r["summary"]
    cols = [("Grade A", (0.12, 0.48, 0.27)), ("URS", (0.78, 0.49, 0.05)), ("Reject", (0.70, 0.15, 0.12)), ("Review", (0.42, 0.25, 0.71))]
    for i, (k, rgb) in enumerate(cols):
        x = 20 * mm + i * 43 * mm
        c.setFillColorRGB(*rgb); c.roundRect(x, y - 16 * mm, 40 * mm, 16 * mm, 2 * mm, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 15); c.drawCentredString(x + 20 * mm, y - 8 * mm, f"{s['pct'][k]:.1f}%")
        c.setFont("Helvetica", 8.5); c.drawCentredString(x + 20 * mm, y - 13 * mm, f"{k} - {s['counts'][k]} of {s['count']}")
    c.setFillColorRGB(0, 0, 0)
    y -= 24 * mm
    c.setFont("Helvetica", 8.5)
    c.drawString(20 * mm, y, r["notice"]); y -= 6 * mm
    c.setFont("Helvetica-Bold", 9); c.drawString(20 * mm, y, "Record fingerprint (SHA-256):"); y -= 4.5 * mm
    c.setFont("Courier", 8); c.drawString(20 * mm, y, report.record_sha256); y -= 4.5 * mm
    for p in r["photos"]:
        c.drawString(20 * mm, y, f"photo {p['sha256']}"); y -= 4 * mm
    qr = QrCodeWidget(f"EKMAAP:{report.record_sha256}")
    b = qr.getBounds(); d = Drawing(32 * mm, 32 * mm, transform=[32 * mm / (b[2] - b[0]), 0, 0, 32 * mm / (b[3] - b[1]), 0, 0]); d.add(qr)
    renderPDF.draw(d, c, W - 52 * mm, H - 52 * mm)
    y -= 4 * mm
    c.setFont("Helvetica-Bold", 9)
    for i, h in enumerate(["Photo", "#", "Diameter mm", "Condition", "Grade", "Officer reviewed"]):
        c.drawString(20 * mm + [0, 18, 30, 55, 85, 110][i] * mm, y, h)
    y -= 5 * mm
    c.setFont("Helvetica", 8.5)
    photo_no = {p["photo_id"]: i + 1 for i, p in enumerate(r["photos"])}
    for o in r["onions"]:
        if y < 20 * mm:
            c.showPage(); y = H - 20 * mm; c.setFont("Helvetica", 8.5)
        vals = [str(photo_no.get(o["photo_id"], "?")), str(o["idx"]), "-" if o["diameter_mm"] is None else f"{o['diameter_mm']:.1f}",
                o["condition"], o["grade"], "yes" if o["officer_reviewed"] else ""]
        for i, v in enumerate(vals):
            c.drawString(20 * mm + [0, 18, 30, 55, 85, 110][i] * mm, y, v)
        y -= 4.6 * mm
    c.drawString(20 * mm, 12 * mm, "Officer signature: ____________________     Farmer signature: ____________________")
    # annotated photos, one per page
    from ..models import Analysis, Photo
    for p in r["photos"]:
        photo = db.get(Photo, p["photo_id"])
        a = db.get(Analysis, p["analysis_id"])
        if not photo or not a:
            continue
        img = decode(storage.read(photo.storage_path))
        view = {"onions": [{"polygon": o_.polygon, "centroid": o_.centroid, "idx": o_.idx, "diameter_mm": o_.diameter_mm,
                            "grade": next((x["grade"] for x in r["onions"] if x["photo_id"] == p["photo_id"] and x["idx"] == o_.idx), o_.grade),
                            "label": next((x["label"] for x in r["onions"] if x["photo_id"] == p["photo_id"] and x["idx"] == o_.idx), o_.label)}
                           for o_ in a.detections]}
        ann = annotate(img, view)
        scale = min(1.0, 1600 / max(ann.shape[:2]))
        ann = cv2.resize(ann, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
        c.showPage()
        c.setFont("Helvetica-Bold", 11); c.drawString(20 * mm, H - 15 * mm, f"Photo {photo_no[p['photo_id']]}  (SHA-256 {p['sha256'][:16]}...)")
        iw, ih = ann.shape[1], ann.shape[0]
        maxw, maxh = W - 30 * mm, H - 40 * mm
        k = min(maxw / iw, maxh / ih)
        c.drawImage(ImageReader(io.BytesIO(jpg.tobytes())), 15 * mm, H - 22 * mm - ih * k, iw * k, ih * k)
    c.save()
    return buf.getvalue()
