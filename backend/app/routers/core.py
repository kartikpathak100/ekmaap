"""Auth, centres, users, farmers, rule sets, health."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ekmaap import __version__
from ekmaap.rules import RuleSet as Rules

from ..audit import log
from ..db import get_db
from ..models import Centre, Farmer, RuleSet, User
from ..security import current_user, hash_password, make_token, require, verify_password
from ..services.engine import clear_cache

router = APIRouter()


def row(o, *fields):
    return {f: (str(v) if isinstance(v := getattr(o, f), uuid.UUID) else v.isoformat() if isinstance(v, datetime) else v) for f in fields}


# ------------------------------------------------------------------ health / auth
@router.get("/health", tags=["system"])
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"ok": True, "ekmaap": __version__}


class Login(BaseModel):
    login: str
    password: str


@router.post("/auth/login", tags=["auth"])
def login(body: Login, db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.login == body.login))
    if not u or not u.active or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Wrong login or password")
    return {"token": make_token(u), "user": row(u, "id", "name", "role", "centre_id")}


@router.get("/auth/me", tags=["auth"])
def me(user: User = Depends(current_user)):
    return row(user, "id", "name", "login", "role", "centre_id")


# ------------------------------------------------------------------ centres / users / farmers
class CentreIn(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    name: str
    district: str | None = None
    state: str | None = None
    lat: float | None = None
    lon: float | None = None


@router.post("/centres", status_code=201, tags=["admin"])
def create_centre(body: CentreIn, db: Session = Depends(get_db), user: User = Depends(require("admin"))):
    c = Centre(**body.model_dump())
    db.add(c)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(409, "Centre code already exists")
    log(db, user, "create", "centre", c.id); db.commit()
    return row(c, "id", "code", "name", "district", "state", "lat", "lon")


@router.get("/centres", tags=["admin"])
def list_centres(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return [row(c, "id", "code", "name", "district", "state") for c in db.scalars(select(Centre).order_by(Centre.code))]


class UserIn(BaseModel):
    name: str
    login: str
    password: str = Field(min_length=8)
    role: str = Field(pattern="^(officer|supervisor|labeller|admin)$")
    centre_id: uuid.UUID | None = None


@router.post("/users", status_code=201, tags=["admin"])
def create_user(body: UserIn, db: Session = Depends(get_db), user: User = Depends(require("admin"))):
    u = User(name=body.name, login=body.login, password_hash=hash_password(body.password), role=body.role, centre_id=body.centre_id)
    db.add(u)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(409, "Login already exists")
    log(db, user, "create", "user", u.id, role=u.role); db.commit()
    return row(u, "id", "name", "login", "role", "centre_id")


@router.get("/users", tags=["admin"])
def list_users(db: Session = Depends(get_db), user: User = Depends(require("admin", "supervisor"))):
    return [row(u, "id", "name", "login", "role", "centre_id", "active") for u in db.scalars(select(User).order_by(User.name))]


class FarmerIn(BaseModel):
    name: str
    phone: str | None = None
    village: str | None = None
    district: str | None = None
    consent: bool = Field(description="Farmer agreed to their name/phone being stored")


@router.post("/farmers", status_code=201, tags=["admin"])
def create_farmer(body: FarmerIn, db: Session = Depends(get_db), user: User = Depends(require("officer", "supervisor"))):
    if not body.consent:
        raise HTTPException(422, "Record the farmer's consent before storing personal data")
    f = Farmer(name=body.name, phone=body.phone, village=body.village, district=body.district, consent_at=datetime.now(timezone.utc))
    db.add(f); db.flush(); log(db, user, "create", "farmer", f.id); db.commit()
    return row(f, "id", "name", "village", "district")


@router.get("/farmers", tags=["admin"])
def list_farmers(q: str = "", db: Session = Depends(get_db), user: User = Depends(current_user)):
    st = select(Farmer).order_by(Farmer.name).limit(50)
    if q:
        st = st.where(Farmer.name.ilike(f"%{q}%"))
    return [row(f, "id", "name", "village", "district") for f in db.scalars(st)]


# ------------------------------------------------------------------ rule sets
class RuleSetIn(BaseModel):
    params: Rules
    source_note: str | None = None
    activate: bool = False


def rs_out(r: RuleSet):
    return row(r, "id", "name", "version", "params", "source_note", "is_active", "created_at")


@router.get("/rulesets", tags=["rule sets"])
def list_rule_sets(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return [rs_out(r) for r in db.scalars(select(RuleSet).order_by(RuleSet.created_at.desc()))]


@router.get("/rulesets/active", tags=["rule sets"])
def get_active_rule_set(db: Session = Depends(get_db), user: User = Depends(current_user)):
    r = db.scalar(select(RuleSet).where(RuleSet.is_active))
    if not r:
        raise HTTPException(404, "No active rule set")
    return rs_out(r)


@router.post("/rulesets", status_code=201, tags=["rule sets"])
def create_rule_set(body: RuleSetIn, db: Session = Depends(get_db), user: User = Depends(require("supervisor"))):
    r = RuleSet(name=body.params.name, version=body.params.version, params=body.params.model_dump(), source_note=body.source_note, created_by=user.id)
    db.add(r)
    try:
        db.flush()
    except IntegrityError:
        raise HTTPException(409, "That version already exists - rule sets are never edited, create a new version")
    log(db, user, "create", "rule_set", r.id, version=r.version)
    if body.activate:
        _activate(db, r, user)
    db.commit()
    return rs_out(r)


def _activate(db, r, user):
    db.execute(update(RuleSet).where(RuleSet.is_active).values(is_active=False))
    db.flush()
    r.is_active = True
    log(db, user, "activate", "rule_set", r.id, version=r.version)


@router.post("/rulesets/{rule_set_id}/activate", tags=["rule sets"])
def activate_rule_set(rule_set_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(require("supervisor"))):
    r = db.get(RuleSet, rule_set_id)
    if not r:
        raise HTTPException(404, "Rule set not found")
    _activate(db, r, user); db.commit(); clear_cache()
    return rs_out(r)
