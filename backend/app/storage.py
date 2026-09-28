"""Content-addressed file store: data/store/ab/cd/<sha256><ext>. Same bytes are stored once."""
import hashlib
from pathlib import Path

from .config import settings


def put(data: bytes, ext: str) -> tuple[str, str]:
    sha = hashlib.sha256(data).hexdigest()
    rel = f"{sha[:2]}/{sha[2:4]}/{sha}{ext}"
    p = settings.data_dir / rel
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    return sha, rel


def path(rel: str) -> Path:
    return settings.data_dir / rel


def read(rel: str) -> bytes:
    return path(rel).read_bytes()
