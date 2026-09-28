# API endpoints

Base path `/api/v1`. Interactive docs (try every call in the browser) at `/docs` when the server runs. The machine-readable spec is [`openapi.json`](openapi.json). Both files are generated from the code by `python scripts/dump_api.py`.

## Authentication and roles

`POST /auth/login` returns a bearer token (valid 12 h by default). Send it as `Authorization: Bearer <token>`.

| Role | Can do |
| --- | --- |
| officer | create lots and farmers, upload lot photos, review onions, issue reports, open disputes |
| supervisor | everything an officer can, plus create/activate rule sets, register/activate models, run evaluations, assign dataset splits, resolve disputes |
| labeller | upload dataset photos, label them (PUT annotations, prelabel, COCO import), export the dataset |
| admin | all of the above, plus centres and users |

`/health`, `/auth/login` and both `/verify` endpoints are public, so a farmer or trader can check a report without an account.

## The three main flows

**Grading a lot (officer)**
1. `POST /lots` creates the lot.
2. `POST /lots/{id}/photos` takes one photo of the sample on the mat. The pipeline runs immediately; the response lists every onion with its polygon, size, condition, grade and whether it **needs review**.
3. `POST /detections/{id}/reviews` records the officer's decision on each flagged onion (and any other onion they want to correct).
4. `GET /lots/{id}/grade` previews the result with reviews applied, using the active rule set.
5. `POST /lots/{id}/reports` freezes the result into a report with a SHA-256 fingerprint. It is refused (409) while any onion still needs review.
6. `GET /reports/{id}/pdf` returns the PDF with a QR code of the fingerprint.

**Checking a report (anyone)**
- `GET /verify/{sha256}` answers whether a report with that fingerprint was issued, and whether it is still valid or was superseded.
- `POST /verify` takes the record JSON (and optionally the photos). It recomputes the fingerprints, so a single changed number or pixel shows up.

**Dispute and re-grade**
`POST /reports/{id}/disputes` re-opens the lot. The officer adds photos or reviews, and issuing again creates a new report that **supersedes** the old one; both stay in the database. `PATCH /disputes/{id}` closes the dispute and points to the new report.

**Training data and models (labeller / supervisor)**
`POST /dataset/photos` → `POST /photos/{id}/prelabel?save=true` → correct the labels in CVAT → `POST /dataset/import` → `POST /dataset/split` → `GET /dataset/export` (a zip the training scripts read directly) → train → `POST /models` → `POST /models/{id}/activate` → `POST /evaluations`.

## All endpoints

<!-- ENDPOINTS:START -->
| Group | Method | Path | What it does |
| --- | --- | --- | --- |
| system | `GET` | `/api/v1/health` | Health |
| auth | `POST` | `/api/v1/auth/login` | Login |
| auth | `GET` | `/api/v1/auth/me` | Me |
| admin | `GET` | `/api/v1/centres` | List Centres |
| admin | `POST` | `/api/v1/centres` | Create Centre |
| admin | `GET` | `/api/v1/farmers` | List Farmers |
| admin | `POST` | `/api/v1/farmers` | Create Farmer |
| admin | `GET` | `/api/v1/users` | List Users |
| admin | `POST` | `/api/v1/users` | Create User |
| rule sets | `GET` | `/api/v1/rulesets` | List Rule Sets |
| rule sets | `POST` | `/api/v1/rulesets` | Create Rule Set |
| rule sets | `GET` | `/api/v1/rulesets/active` | Get Active Rule Set |
| rule sets | `POST` | `/api/v1/rulesets/{rule_set_id}/activate` | Activate Rule Set |
| lots | `GET` | `/api/v1/lots` | List Lots |
| lots | `POST` | `/api/v1/lots` | Create Lot |
| lots | `GET` | `/api/v1/lots/{lot_id}` | Get Lot Detail |
| lots | `GET` | `/api/v1/lots/{lot_id}/grade` | Lot result right now: latest analysis of every photo, officer reviews applied, active rule set. |
| lots | `POST` | `/api/v1/lots/{lot_id}/photos` | Upload one photo of the lot's sample (on the printed mat). |
| photos | `POST` | `/api/v1/detections/{detection_id}/reviews` | Officer's decision on one onion (confirms or corrects the model). |
| photos | `GET` | `/api/v1/photos/{photo_id}` | Photo Meta |
| photos | `POST` | `/api/v1/photos/{photo_id}/analyse` | Run the CURRENT active models again (new analysis row; old ones are kept). |
| photos | `GET` | `/api/v1/photos/{photo_id}/analysis` | Photo Latest Analysis |
| photos | `GET` | `/api/v1/photos/{photo_id}/annotated.jpg` | Photo Annotated |
| photos | `GET` | `/api/v1/photos/{photo_id}/image` | Photo Image |
| reports | `POST` | `/api/v1/lots/{lot_id}/reports` | Freeze the lot result into a fingerprinted report. |
| reports | `GET` | `/api/v1/reports/{report_id}` | Report Json |
| reports | `GET` | `/api/v1/reports/{report_id}/pdf` | Report Pdf |
| disputes | `GET` | `/api/v1/disputes` | List Disputes |
| disputes | `PATCH` | `/api/v1/disputes/{dispute_id}` | Resolve Dispute |
| disputes | `POST` | `/api/v1/reports/{report_id}/disputes` | Open Dispute |
| verify (public) | `POST` | `/api/v1/verify` | Public: recompute the fingerprint of a record (and optionally of photos) and compare with what was issued. |
| verify (public) | `GET` | `/api/v1/verify/{record_sha256}` | Public: is there an issued report with this fingerprint? (What the QR code on the PDF encodes.). |
| dataset | `GET` | `/api/v1/dataset/export` | ZIP with images/, annotations.json (COCO, categories = conditions, per-image split) and ruler.csv - exactly what the scripts/train_*.py and scripts/evaluate.py expect. |
| dataset | `POST` | `/api/v1/dataset/import` | Import corrected labels from CVAT / Label Studio (COCO). |
| dataset | `GET` | `/api/v1/dataset/photos` | List Dataset |
| dataset | `POST` | `/api/v1/dataset/photos` | Add a photo for training/evaluation (not linked to a lot). |
| dataset | `POST` | `/api/v1/dataset/split` | Assign train/val/test by photo (or by split_group). |
| dataset | `GET` | `/api/v1/photos/{photo_id}/annotations` | Get Annotations |
| dataset | `PUT` | `/api/v1/photos/{photo_id}/annotations` | Replace ALL ground-truth labels of a photo (polygons in original photo pixels). |
| dataset | `POST` | `/api/v1/photos/{photo_id}/prelabel` | Suggest outlines + conditions with the active models (faster than drawing). |
| models | `GET` | `/api/v1/evaluations` | List Evaluations |
| models | `POST` | `/api/v1/evaluations` | Evaluate the ACTIVE models + rule set on labelled photos of a split (synchronous; keep test sets modest). |
| models | `GET` | `/api/v1/models` | List Models |
| models | `POST` | `/api/v1/models` | Register a trained model file (from scripts/train_*.py). |
| models | `POST` | `/api/v1/models/{model_id}/activate` | Activate Model |
<!-- ENDPOINTS:END -->

## Error codes you will see

| Code | Meaning |
| --- | --- |
| 401 | missing, wrong or expired token |
| 403 | your role cannot do this |
| 404 | id not found |
| 409 | conflict: duplicate code/version, lot already reported, onions still need review, no active rule set |
| 413 / 415 / 422 | photo too large / not an image / invalid data |
| 503 | an active model file cannot be loaded: activate another version |
