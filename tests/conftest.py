"""Tests run against a real PostgreSQL. Set EKMAAP_TEST_DATABASE_URL, e.g.
postgresql+psycopg://postgres@localhost:5432/ekmaap_test  (the database is wiped and rebuilt from db/schema.sql)."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_DB = os.environ.get("EKMAAP_TEST_DATABASE_URL")
os.environ["EKMAAP_DATABASE_URL"] = TEST_DB or "postgresql+psycopg://invalid/none"
os.environ.setdefault("EKMAAP_DATA_DIR", tempfile.mkdtemp(prefix="ekmaap_store_"))
os.environ.setdefault("EKMAAP_SECRET_KEY", "test-secret")


@pytest.fixture(scope="session")
def db_url():
    if not TEST_DB:
        pytest.skip("EKMAAP_TEST_DATABASE_URL not set")
    return TEST_DB


@pytest.fixture(scope="session")
def fresh_db(db_url):
    import sqlalchemy as sa
    eng = sa.create_engine(db_url)
    with eng.begin() as c:
        c.exec_driver_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        c.exec_driver_sql((ROOT / "db" / "schema.sql").read_text())
    eng.dispose()
    return db_url
