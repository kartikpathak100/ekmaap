from pathlib import Path


def test_schema_sql_matches_models():
    from backend.app.dump_schema import render
    sql = (Path(__file__).resolve().parents[1] / "db" / "schema.sql").read_text()
    assert sql == render(), "db/schema.sql is stale: run  python -m backend.app.dump_schema"
