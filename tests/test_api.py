"""End-to-end API test on PostgreSQL: setup -> lot -> photo -> review -> report -> verify -> dispute,
plus dataset labels, export/import, model registry and evaluation."""
import io
import json
import zipfile

import cv2
import joblib
import numpy as np
import pytest

from ekmaap import synth


@pytest.fixture(scope="module")
def client(fresh_db):
    import os
    os.environ["EKMAAP_ADMIN_LOGIN"] = "admin"
    os.environ["EKMAAP_ADMIN_PASSWORD"] = "admin-password"
    from backend.app.config import settings
    settings.admin_login, settings.admin_password = "admin", "admin-password"
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as c:
        yield c


def jpg(seed, **kw):
    img, onions = synth.make_photo(seed=seed, **kw)
    return cv2.imencode(".jpg", img)[1].tobytes(), img, onions


def auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def login(c, who, pw):
    r = c.post("/api/v1/auth/login", json={"login": who, "password": pw})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def test_full_grading_workflow(client):
    c = client
    assert c.get("/api/v1/health").json()["ok"]
    assert c.get("/api/v1/lots").status_code == 401
    A = auth(login(c, "admin", "admin-password"))
    centre = c.post("/api/v1/centres", json={"code": "LSG-01", "name": "Lasalgaon demo centre", "district": "Nashik"}, headers=A).json()
    for login_, role in (("officer1", "officer"), ("sup1", "supervisor"), ("lab1", "labeller")):
        r = c.post("/api/v1/users", json={"name": login_, "login": login_, "password": "password123", "role": role, "centre_id": centre["id"]}, headers=A)
        assert r.status_code == 201, r.text
    O, S = auth(login(c, "officer1", "password123")), auth(login(c, "sup1", "password123"))
    # officers cannot create rule sets; supervisors can
    assert c.post("/api/v1/rulesets", json={"params": {}}, headers=O).status_code == 403
    r = c.post("/api/v1/rulesets", json={"params": {"version": "test-v1"}, "source_note": "demo values", "activate": True}, headers=S)
    assert r.status_code == 201 and r.json()["is_active"], r.text
    assert c.post("/api/v1/rulesets", json={"params": {"version": "test-v1"}}, headers=S).status_code == 409
    farmer = c.post("/api/v1/farmers", json={"name": "Test Farmer", "consent": True}, headers=O).json()
    lot = c.post("/api/v1/lots", json={"lot_code": "LOT-0001", "centre_id": centre["id"], "farmer_id": farmer["id"], "variety": "red"}, headers=O).json()

    data, img, onions = jpg(11, touching_pairs=0)
    r = c.post(f"/api/v1/lots/{lot['id']}/photos", files={"photo": ("s.jpg", data, "image/jpeg")}, headers=O)
    assert r.status_code == 201, r.text
    an = r.json()
    assert an["scale_found"] and len(an["detections"]) == len(onions)
    assert c.post(f"/api/v1/lots/{lot['id']}/photos", files={"photo": ("x.txt", b"hello", "text/plain")}, headers=O).status_code == 415
    assert c.get(f"/api/v1/photos/{an['photo_id']}/annotated.jpg", headers=O).headers["content-type"] == "image/jpeg"

    # every onion flagged for review must be decided before a report can be issued
    pending = [d for d in an["detections"] if d["grade"] == "Review"]
    if pending:
        r = c.post(f"/api/v1/lots/{lot['id']}/reports", json={}, headers=O)
        assert r.status_code == 409
    for d in pending:
        assert c.post(f"/api/v1/detections/{d['id']}/reviews", json={"condition": "sound", "note": "checked by hand"}, headers=O).status_code == 201
    # one explicit correction
    d0 = an["detections"][0]
    c.post(f"/api/v1/detections/{d0['id']}/reviews", json={"condition": "rotten", "note": "soft base"}, headers=O)
    g = c.get(f"/api/v1/lots/{lot['id']}/grade", headers=O).json()
    assert g["summary"]["count"] == len(onions) and g["summary"]["counts"]["Review"] == 0
    assert next(o for o in g["onions"] if o["detection_id"] == d0["id"])["grade"] == "Reject"

    r = c.post(f"/api/v1/lots/{lot['id']}/reports", json={}, headers=O)
    assert r.status_code == 201, r.text
    rep = r.json()
    # locked after reporting
    assert c.post(f"/api/v1/lots/{lot['id']}/photos", files={"photo": ("s.jpg", data, "image/jpeg")}, headers=O).status_code == 409
    pdf = c.get(f"/api/v1/reports/{rep['id']}/pdf", headers=O)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    # public verification: exact record passes, one changed number fails, photo hash matches
    stored = c.get(f"/api/v1/reports/{rep['id']}", headers=O).json()
    v = c.post("/api/v1/verify", files={"record": ("r.json", json.dumps(stored).encode(), "application/json"),
                                        "photos": ("s.jpg", data, "image/jpeg")}).json()
    assert v["issued"] and v["unchanged"] and v["fingerprint"] == rep["record_sha256"] and v["photos"][0]["in_record"]
    tampered = json.loads(json.dumps(stored["record"]))
    tampered["summary"]["pct"]["Grade A"] += 10
    v2 = c.post("/api/v1/verify", files={"record": ("r.json", json.dumps(tampered).encode(), "application/json")}).json()
    assert not v2["issued"]
    assert c.get(f"/api/v1/verify/{rep['record_sha256']}").json()["found"]

    # dispute -> re-grade -> new report supersedes old -> dispute resolved
    dsp = c.post(f"/api/v1/reports/{rep['id']}/disputes", json={"raised_by_name": "Test Farmer", "raised_by_role": "farmer", "reason": "Onion 1 is not rotten"}, headers=O).json()
    c.post(f"/api/v1/detections/{d0['id']}/reviews", json={"condition": "sound", "note": "re-checked with farmer"}, headers=S)
    rep2 = c.post(f"/api/v1/lots/{lot['id']}/reports", json={}, headers=S).json()
    assert rep2["record"]["supersedes"] == rep["record_sha256"]
    assert c.get(f"/api/v1/verify/{rep['record_sha256']}").json()["status"] == "superseded"
    r = c.patch(f"/api/v1/disputes/{dsp['id']}", json={"status": "regraded", "resolution_note": "onion 1 sound", "new_report_id": rep2["id"]}, headers=S)
    assert r.status_code == 200 and r.json()["status"] == "regraded"


def test_dataset_models_and_evaluation(client):
    c = client
    A = auth(login(c, "admin", "admin-password"))
    L, S = auth(login(c, "lab1", "password123")), auth(login(c, "sup1", "password123"))
    ids = []
    truth = {}
    for k in range(6):
        data, img, onions = jpg(100 + k, touching_pairs=0)
        p = c.post("/api/v1/dataset/photos", files={"photo": (f"d{k}.jpg", data, "image/jpeg")}, data={"split_group": f"day{k % 3}"}, headers=L)
        assert p.status_code == 201, p.text
        pid = p.json()["id"]; ids.append(pid)
        items = [{"polygon": o.polygon_img.round(1).tolist(), "condition": o.condition, "ruler_mm": round(o.d_mm, 1)} for o in onions]
        truth[pid] = len(items)
        assert c.put(f"/api/v1/photos/{pid}/annotations", json=items, headers=L).json()["annotations"] == len(items)
    # duplicate upload refused
    assert c.post("/api/v1/dataset/photos", files={"photo": ("again.jpg", jpg(100, touching_pairs=0)[0], "image/jpeg")}, headers=L).status_code == 409
    # prelabel suggestions exist
    sug = c.post(f"/api/v1/photos/{ids[0]}/prelabel", headers=L).json()
    assert sug["scale_found"] and len(sug["suggestions"]) > 0
    split = c.post("/api/v1/dataset/split", json={"train": 0.34, "val": 0.33, "reassign": True}, headers=S).json()["assigned"]
    assert sum(split.values()) == 6

    # export -> zip in the training-script format -> re-import (round trip)
    z = zipfile.ZipFile(io.BytesIO(c.get("/api/v1/dataset/export", headers=L).content))
    coco = json.loads(z.read("annotations.json"))
    assert len(coco["images"]) == 6 and len(coco["annotations"]) == sum(truth.values())
    assert z.read("ruler.csv").decode().count("\n") == sum(truth.values()) + 1
    r = c.post("/api/v1/dataset/import", files={"coco_file": ("a.json", json.dumps(coco).encode(), "application/json")}, headers=L).json()
    assert r["photos_updated"] == 6 and not r["unmatched_files"]

    # registry: a size calibration + a trained classifier; bad files are refused
    from ekmaap.classifier import LearnedClassifier
    from ekmaap.features import NAMES
    import tempfile, os
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(NAMES))); y = np.array(["sound", "rotten"] * 20)
    clf = LearnedClassifier.train(X, y, np.repeat(np.arange(4), 10))
    f = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False); f.close(); clf.save(f.name)
    r = c.post("/api/v1/models", data={"kind": "classifier", "backend": "sklearn", "version": "clf-test-1"},
               files={"file": ("c.joblib", open(f.name, "rb").read(), "application/octet-stream")}, headers=S)
    assert r.status_code == 201, r.text
    bad = c.post("/api/v1/models", data={"kind": "classifier", "backend": "sklearn", "version": "clf-bad"},
                 files={"file": ("c.joblib", b"not a model", "application/octet-stream")}, headers=S)
    assert bad.status_code == 422
    sc = c.post("/api/v1/models", data={"kind": "size_calibration", "backend": "linear", "version": "size-1"},
                files={"file": ("s.json", json.dumps({"slope": 1.0, "intercept": -0.5}).encode(), "application/json")}, headers=S).json()
    assert c.post(f"/api/v1/models/{sc['id']}/activate", headers=S).json()["is_active"]
    assert c.post(f"/api/v1/models/{sc['id']}/activate", headers=L).status_code == 403

    ev = c.post("/api/v1/evaluations", json={"split": "train"}, headers=S)
    assert ev.status_code == 201, ev.text
    m = ev.json()["metrics"]
    assert m["detection"]["recall"] > 0.8 and m["size"]["n_ruler"] > 0
    assert len(c.get("/api/v1/evaluations", headers=S).json()) == 1
