# EkMaap: onion grading from photos (SIH26031)

Team **StrataCadastre_KP** · Smart India Hackathon 2026 · problem statement SIH26031 (Department of Consumer Affairs).

An officer photographs the onion sample on a printed mat. EkMaap finds every onion, measures it in millimetres, and decides whether it is sound, sprouted, damaged or rotten. It then applies a versioned grading rule set (Grade A / URS / Reject) and issues a report with a SHA-256 fingerprint that anyone can verify. Onions the model is unsure about go to the officer, who decides; every decision is stored.

> **Status: prototype.** No model has been trained on real onions yet. The untrained defaults have not been tuned on real photographs. Next steps: collect and label real photos, train, and measure on held-out photos ([docs/training.md](docs/training.md)). Grade limits in the demo rule set (45 mm / 35 mm) come from 2026 news reports, **not** the official specification.

## Documents

| | |
| --- | --- |
| [System architecture](docs/architecture.md) | components, photo-to-report sequence, pipeline stages, deployment, limits |
| [Database schema](docs/database.md) | 15 tables, ER diagram, rules the schema enforces; DDL in [db/schema.sql](db/schema.sql) |
| [API endpoints](docs/api.md) | 45 endpoints, roles, the main flows; [openapi.json](docs/openapi.json) |
| [How to train](docs/training.md) | photo protocol, labelling in CVAT, training, evaluation, deploying models |

## Repository map

```
ekmaap/      the pipeline (shared by API and training): calibration, segmentation, measure, features,
             classifier, rules, pipeline, dataset, evaluate, coco, synth (synthetic test images)
backend/     FastAPI app: models.py (schema), routers/, services/ (engine, reports), security, storage
frontend/    officer web app (served by the API at /)
scripts/     make_mat, prelabel, train_segmenter, train_classifier, fit_size_calibration, evaluate,
             coco_to_yolo, train_yolo_seg (GPU, optional), seed_demo, make_synthetic, dump_api
db/          schema.sql (generated from backend/app/models.py)
tests/       pytest: pipeline on synthetic photos + full API workflow on PostgreSQL
notebooks/   Colab notebook for the optional GPU model
demo/        the first offline HTML demo (heuristics only, kept for reference)
```

## Run it

**With Docker** (not yet built by us, see "Verified"):
```bash
cp .env.example .env            # set passwords and secret
docker compose up -d --build    # http://localhost:8000  (app)  and  /docs  (API)
python scripts/seed_demo.py --url http://localhost:8000 --admin admin --password <EKMAAP_ADMIN_PASSWORD>
```

**Without Docker:**
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
createdb ekmaap && psql ekmaap -f db/schema.sql
export EKMAAP_DATABASE_URL=postgresql+psycopg://$USER@localhost/ekmaap EKMAAP_SECRET_KEY=dev EKMAAP_ADMIN_LOGIN=admin EKMAAP_ADMIN_PASSWORD=admin-pass-123
uvicorn backend.app.main:app --reload
python scripts/seed_demo.py --url http://localhost:8000 --admin admin --password admin-pass-123
python scripts/make_mat.py      # print ekmaap_mat.pdf at 100 %
```

**Try the training loop without real data** (synthetic, only proves the scripts run):
```bash
make synthetic && make train D=data/synthetic && make evaluate D=data/synthetic
```

**Tests:** `EKMAAP_TEST_DATABASE_URL=postgresql+psycopg://user@localhost/ekmaap_test python -m pytest -q`. The test database is wiped.

## Verified, and not verified

| Ran and passed | Not run yet |
| --- | --- |
| 12 automated tests: pipeline on synthetic photos; full API workflow on PostgreSQL 16 (lot → photo → review → report → PDF → verify, tampering detected, dispute → superseding report, dataset export/import, model registry, evaluation); schema.sql in sync | anything on **real** onions |
| every training script end to end on a 30-photo synthetic set (see docs/training.md for the numbers) | YOLO-seg training/inference (no PyTorch available while building) |
| officer web app driven in Chromium at desktop and 390 px phone width | `docker compose` build (no package-index access while building); a real CVAT export |

## Licence notes

Pipeline, API and app: choose a licence before publishing (MIT or Apache-2.0 are common). The optional `ultralytics` dependency is AGPL-3.0.
