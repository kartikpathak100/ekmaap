"""EkMaap API.  Run:  uvicorn backend.app.main:app --reload   (from the repo root)

Interactive docs at /docs. All endpoints are under /api/v1. The officer web client is served at /.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from ekmaap import __version__

from .config import settings
from .db import SessionLocal
from .models import User
from .routers import core, grading, training
from .security import hash_password

app = FastAPI(title="EkMaap onion grading API", version=__version__,
              description="SIH26031 - photo-based onion grading. Prototype: see docs/ for limits.")
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

for r in (core.router, grading.router, training.router):
    app.include_router(r, prefix="/api/v1")


@app.on_event("startup")
def bootstrap_admin():
    """Create the first admin from EKMAAP_ADMIN_LOGIN / EKMAAP_ADMIN_PASSWORD if there are no users yet."""
    if not (settings.admin_login and settings.admin_password):
        return
    with SessionLocal() as db:
        if db.scalar(select(User).limit(1)) is None:
            db.add(User(name="Admin", login=settings.admin_login, password_hash=hash_password(settings.admin_password), role="admin"))
            db.commit()


frontend = Path(__file__).resolve().parents[2] / "frontend"
if frontend.exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
