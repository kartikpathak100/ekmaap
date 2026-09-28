"""Regenerate docs/openapi.json and the endpoint table in docs/api.md from the running code."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.main import app  # noqa: E402

spec = app.openapi()
(ROOT / "docs" / "openapi.json").write_text(json.dumps(spec, indent=1))
AUTH = {"public": "none", "current_user": "any signed-in user"}
rows = []
for path, ops in spec["paths"].items():
    for method, op in ops.items():
        roles = "none (public)" if "verify" in path or path.endswith("/health") or path.endswith("/auth/login") else "signed in"
        text = " ".join((op.get("description") or op.get("summary") or "").split())
        first = text.split(". ")[0].rstrip(".") + ("." if op.get("description") else "")
        rows.append((op.get("tags", ["-"])[0], method.upper(), path, first, roles))
order = ["system", "auth", "admin", "rule sets", "lots", "photos", "reports", "disputes", "verify (public)", "dataset", "models"]
rows.sort(key=lambda r: (order.index(r[0]) if r[0] in order else 99, r[2], r[1]))
table = "| Group | Method | Path | What it does |\n| --- | --- | --- | --- |\n" + "".join(
    f"| {g} | `{m}` | `{p}` | {d} |\n" for g, m, p, d, _ in rows)
doc = ROOT / "docs" / "api.md"
text = doc.read_text()
start, end = "<!-- ENDPOINTS:START -->", "<!-- ENDPOINTS:END -->"
doc.write_text(text.split(start)[0] + start + "\n" + table + end + text.split(end)[1])
print(f"{len(rows)} endpoints")
