"""Write db/schema.sql from the ORM models:  python -m backend.app.dump_schema"""
from pathlib import Path

from sqlalchemy import create_mock_engine
from sqlalchemy.schema import CreateIndex, CreateTable, SetColumnComment, SetTableComment

from .models import Base

HEADER = """-- EkMaap database schema (PostgreSQL 14+). GENERATED from backend/app/models.py - do not edit by hand.
-- Apply:  psql "$DATABASE_URL" -f db/schema.sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid() on PostgreSQL < 13

"""


def render() -> str:
    out = []
    eng = create_mock_engine("postgresql+psycopg://", lambda sql, *a, **k: None)
    d = eng.dialect
    for t in Base.metadata.sorted_tables:
        out.append(str(CreateTable(t).compile(dialect=d)).strip() + ";\n")
        for ix in sorted(t.indexes, key=lambda i: i.name):
            out.append(str(CreateIndex(ix).compile(dialect=d)).strip() + ";\n")
        if t.comment:
            out.append(str(SetTableComment(t).compile(dialect=d)).strip() + ";\n")
        for c in t.columns:
            if c.comment:
                out.append(str(SetColumnComment(c).compile(dialect=d)).strip() + ";\n")
        out.append("\n")
    return HEADER + "".join(out)


if __name__ == "__main__":
    p = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
    p.write_text(render())
    print("wrote", p)
