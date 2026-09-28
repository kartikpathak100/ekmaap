"""Create a demo centre, users and the demo rule set through the API (run once on a fresh database).

    EKMAAP_ADMIN_LOGIN=admin EKMAAP_ADMIN_PASSWORD=... uvicorn backend.app.main:app   # server creates the admin
    python scripts/seed_demo.py --url http://localhost:8000 --admin admin --password ...

Creates: centre DEMO-01; users officer1 / supervisor1 / labeller1 (password printed); rule set demo-2026-09-28 (active).
The rule set values are DEMO values from 2026 news reports, not the official specification.
"""
import argparse
import secrets

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--admin", required=True)
ap.add_argument("--password", required=True)
a = ap.parse_args()
c = httpx.Client(base_url=a.url.rstrip("/") + "/api/v1", timeout=30)
tok = c.post("/auth/login", json={"login": a.admin, "password": a.password}).raise_for_status().json()["token"]
H = {"Authorization": f"Bearer {tok}"}
centre = c.post("/centres", json={"code": "DEMO-01", "name": "Demo procurement centre"}, headers=H).raise_for_status().json()
pw = secrets.token_urlsafe(9)
for login, role in (("officer1", "officer"), ("supervisor1", "supervisor"), ("labeller1", "labeller")):
    c.post("/users", json={"name": login, "login": login, "password": pw, "role": role, "centre_id": centre["id"]}, headers=H).raise_for_status()
sup = c.post("/auth/login", json={"login": "supervisor1", "password": pw}).json()["token"]
c.post("/rulesets", json={"params": {}, "source_note": "Demo values from 2026 news reports - NOT official", "activate": True},
       headers={"Authorization": f"Bearer {sup}"}).raise_for_status()
print(f"centre DEMO-01, users officer1 / supervisor1 / labeller1, password: {pw}")
