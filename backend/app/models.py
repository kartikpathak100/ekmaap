"""Database schema (PostgreSQL). Single source of truth: db/schema.sql is generated from this file
by ``python -m backend.app.dump_schema``.

Design rules
  * Nothing that decided a grade is ever overwritten: analyses, rule sets, model versions and issued
    reports are append-only; a re-grade creates new rows and a new report that supersedes the old.
  * Photos are stored once, by SHA-256; the database keeps the hash and the storage path.
  * Ground-truth labels for training (``annotations``) are separate from model output (``detections``)
    and from officer corrections (``detection_reviews``).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, Numeric,
                        String, Text, UniqueConstraint, func, text)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def pk():
    return mapped_column(primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))


def created():
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def one_of(col: str, values) -> CheckConstraint:
    vals = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    return CheckConstraint(f"{col} IN ({vals})", name=f"ck_{col}")


ROLES = ("officer", "supervisor", "labeller", "admin")
CONDITIONS = ("sound", "sprouted", "damaged", "rotten")
GRADES = ("Grade A", "URS", "Reject", "Review")


class Centre(Base):
    __tablename__ = "centres"
    __table_args__ = {"comment": "Procurement centres (NAFED / NCCF / APMC sites)."}
    id: Mapped[uuid.UUID] = pk()
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    district: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str | None] = mapped_column(String(100))
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = created()


class User(Base):
    __tablename__ = "users"
    __table_args__ = (one_of("role", ROLES), {"comment": "Officers, supervisors, labellers, admins. login = phone or email."})
    id: Mapped[uuid.UUID] = pk()
    name: Mapped[str] = mapped_column(String(200))
    login: Mapped[str] = mapped_column(String(120), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20))
    centre_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("centres.id"))
    active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    created_at: Mapped[datetime] = created()


class Farmer(Base):
    __tablename__ = "farmers"
    __table_args__ = {"comment": "Sellers. Minimal personal data; consent time recorded (DPDP Act 2023)."}
    id: Mapped[uuid.UUID] = pk()
    name: Mapped[str] = mapped_column(String(200))
    phone: Mapped[str | None] = mapped_column(String(20))
    village: Mapped[str | None] = mapped_column(String(120))
    district: Mapped[str | None] = mapped_column(String(100))
    consent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created()


class RuleSet(Base):
    __tablename__ = "rule_sets"
    __table_args__ = (Index("uq_rule_sets_one_active", "is_active", unique=True, postgresql_where=text("is_active")),
                      {"comment": "Versioned grading rules (size limits, condition -> grade). Never edited after use; make a new version."})
    id: Mapped[uuid.UUID] = pk()
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(64), unique=True)
    params: Mapped[dict] = mapped_column(JSONB)
    source_note: Mapped[str | None] = mapped_column(Text, comment="Where the limits come from (circular no., date).")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = created()


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (one_of("kind", ("segmenter", "classifier", "size_calibration")),
                      one_of("backend", ("classical", "pixel_forest", "yolo_seg", "heuristic", "sklearn", "linear")),
                      Index("uq_model_versions_one_active_per_kind", "kind", unique=True, postgresql_where=text("is_active")),
                      {"comment": "Trained model files and their test-split metrics. One active version per kind."})
    id: Mapped[uuid.UUID] = pk()
    kind: Mapped[str] = mapped_column(String(20))
    backend: Mapped[str] = mapped_column(String(20))
    version: Mapped[str] = mapped_column(String(64), unique=True)
    file_sha256: Mapped[str | None] = mapped_column(String(64))
    storage_path: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    notes: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = created()


class Lot(Base):
    __tablename__ = "lots"
    __table_args__ = (one_of("status", ("open", "graded", "reported", "disputed", "closed")),
                      {"comment": "A consignment brought to a centre for grading."})
    id: Mapped[uuid.UUID] = pk()
    lot_code: Mapped[str] = mapped_column(String(64), unique=True)
    centre_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("centres.id"))
    officer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    farmer_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("farmers.id"))
    variety: Mapped[str | None] = mapped_column(String(100))
    declared_weight_kg: Mapped[float | None] = mapped_column(Numeric(12, 2))
    status: Mapped[str] = mapped_column(String(20), default="open", server_default="open")
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = created()
    photos: Mapped[list["Photo"]] = relationship(back_populates="lot", order_by="Photo.created_at")


class Photo(Base):
    __tablename__ = "photos"
    __table_args__ = (one_of("purpose", ("lot", "dataset")), one_of("split", ("unassigned", "train", "val", "test")),
                      Index("uq_photos_dataset_sha", "sha256", unique=True, postgresql_where=text("purpose = 'dataset'")),
                      Index("ix_photos_lot", "lot_id"),
                      {"comment": "Every uploaded photo: of a lot (grading) or for the training dataset."})
    id: Mapped[uuid.UUID] = pk()
    lot_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("lots.id", ondelete="CASCADE"))
    purpose: Mapped[str] = mapped_column(String(10))
    sha256: Mapped[str] = mapped_column(String(64))
    storage_path: Mapped[str] = mapped_column(Text)
    original_name: Mapped[str | None] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(50))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    split: Mapped[str] = mapped_column(String(12), default="unassigned", server_default="unassigned")
    split_group: Mapped[str | None] = mapped_column(String(64), comment="e.g. capture day; splits keep a group together")
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = created()
    lot: Mapped[Lot | None] = relationship(back_populates="photos")


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (Index("ix_analyses_photo", "photo_id", "created_at"),
                      {"comment": "One run of the pipeline on one photo, with the exact models and rule set used."})
    id: Mapped[uuid.UUID] = pk()
    photo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("photos.id", ondelete="CASCADE"))
    segmenter_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id"))
    classifier_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id"))
    size_cal_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id"))
    rule_set_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rule_sets.id"))
    scale_found: Mapped[bool] = mapped_column(Boolean)
    calibration: Mapped[dict | None] = mapped_column(JSONB)
    summary: Mapped[dict] = mapped_column(JSONB)
    timing_ms: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created()
    detections: Mapped[list["Detection"]] = relationship(back_populates="analysis", order_by="Detection.idx", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"
    __table_args__ = (one_of("condition", CONDITIONS), one_of("grade", GRADES), UniqueConstraint("analysis_id", "idx"),
                      {"comment": "One onion found by the pipeline. Polygon in original photo pixels."})
    id: Mapped[uuid.UUID] = pk()
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"))
    idx: Mapped[int] = mapped_column(Integer)
    polygon: Mapped[list] = mapped_column(JSONB)
    centroid: Mapped[list] = mapped_column(JSONB)
    diameter_mm: Mapped[float | None] = mapped_column(Float)
    width_mm: Mapped[float | None] = mapped_column(Float)
    condition: Mapped[str] = mapped_column(String(10))
    proba: Mapped[dict] = mapped_column(JSONB)
    grade: Mapped[str] = mapped_column(String(10))
    label: Mapped[str] = mapped_column(String(20))
    needs_review: Mapped[bool] = mapped_column(Boolean)
    reasons: Mapped[list] = mapped_column(JSONB)
    features: Mapped[dict] = mapped_column(JSONB)
    analysis: Mapped[Analysis] = relationship(back_populates="detections")
    reviews: Mapped[list["DetectionReview"]] = relationship(order_by="DetectionReview.created_at")


class DetectionReview(Base):
    __tablename__ = "detection_reviews"
    __table_args__ = (one_of("condition", CONDITIONS),
                      {"comment": "Officer corrections at grading time. Latest review wins; history kept."})
    id: Mapped[uuid.UUID] = pk()
    detection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("detections.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    condition: Mapped[str] = mapped_column(String(10))
    diameter_mm: Mapped[float | None] = mapped_column(Float, comment="Set only if the officer measured it")
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created()


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (one_of("condition", CONDITIONS), one_of("source", ("manual", "prelabel", "import", "review")),
                      {"comment": "Ground-truth labels for TRAINING and EVALUATION (polygon + condition, optional ruler size)."})
    id: Mapped[uuid.UUID] = pk()
    photo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("photos.id", ondelete="CASCADE"), index=True)
    polygon: Mapped[list] = mapped_column(JSONB)
    condition: Mapped[str] = mapped_column(String(10))
    ruler_mm: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(10))
    labeller_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = created()


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (one_of("status", ("valid", "superseded")),
                      {"comment": "Issued quality reports. record is exactly what was fingerprinted; never modified."})
    id: Mapped[uuid.UUID] = pk()
    lot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("lots.id"), index=True)
    record: Mapped[dict] = mapped_column(JSONB)
    canonical: Mapped[str] = mapped_column(Text, comment="The exact UTF-8 text that was hashed (sorted keys, no spaces).")
    record_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(12), default="valid", server_default="valid")
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("reports.id"))
    issued_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    issued_at: Mapped[datetime] = created()


class Dispute(Base):
    __tablename__ = "disputes"
    __table_args__ = (one_of("raised_by_role", ("farmer", "officer", "trader", "other")),
                      one_of("status", ("open", "regraded", "upheld", "rejected")),
                      {"comment": "A challenge to an issued report and how it was resolved."})
    id: Mapped[uuid.UUID] = pk()
    report_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reports.id"), index=True)
    raised_by_name: Mapped[str] = mapped_column(String(200))
    raised_by_role: Mapped[str] = mapped_column(String(10))
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(10), default="open", server_default="open")
    resolution_note: Mapped[str | None] = mapped_column(Text)
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    new_report_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("reports.id"))
    created_at: Mapped[datetime] = created()
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Evaluation(Base):
    __tablename__ = "evaluations"
    __table_args__ = {"comment": "Test-split results of a model combination. The only numbers to quote."}
    id: Mapped[uuid.UUID] = pk()
    segmenter_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id"))
    classifier_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id"))
    size_cal_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id"))
    rule_set_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rule_sets.id"))
    split: Mapped[str] = mapped_column(String(12))
    n_photos: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSONB)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = created()


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = {"comment": "Who did what, when. Append-only."}
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(64))
    entity: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    detail: Mapped[dict | None] = mapped_column(JSONB)
    at: Mapped[datetime] = created()
