"""Synthetic onion photos with exact ground truth - for TESTS ONLY.

Draws red onions (sound, small, sprouted, cut, rotten) on a printed ArUco mat,
then photographs it with a simulated tilted phone (perspective, uneven light,
blur, noise, JPEG). Every onion's polygon, condition and true diameter are
known, so the code can be checked end to end.

These images are drawn by our own code. Good scores on them show the code
works; they say nothing about real onions. Onion height is not simulated
(the mat and onions are flat), so real photos will show a size bias this
generator cannot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import ARUCO_DICT, DEFAULT_MAT, MatSpec

PAPER = (228, 234, 236)  # BGR


@dataclass
class SynthOnion:
    cx_mm: float
    cy_mm: float
    d_mm: float          # true widest diameter
    aspect: float        # minor / major
    angle: float         # radians
    condition: str       # sound | sprouted | damaged | rotten
    polygon_img: np.ndarray | None = None  # in the final photo, px


def _render_mat(spec: MatSpec, ppm: float) -> np.ndarray:
    W, H = int(spec.width_mm * ppm), int(spec.height_mm * ppm)
    mat = np.zeros((H, W, 3), np.uint8)
    mat[:] = PAPER
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    s = int(round(spec.marker_mm * ppm))
    for mid, (x, y) in spec.positions.items():
        m = cv2.aruco.generateImageMarker(d, mid, s)
        x0, y0 = int(round(x * ppm)), int(round(y * ppm))
        mat[y0:y0 + s, x0:x0 + s] = m[..., None]
    return mat


def _place(rng, spec: MatSpec, n: int, touching_pairs: int, small: int, defects: dict) -> list[SynthOnion]:
    conds = ["sprouted"] * defects.get("sprouted", 0) + ["damaged"] * defects.get("damaged", 0) + ["rotten"] * defects.get("rotten", 0)
    conds += ["small"] * small
    conds += ["sound"] * max(0, n - len(conds))
    rng.shuffle(conds)
    keep_out = [(x - 6, y - 6, x + spec.marker_mm + 6, y + spec.marker_mm + 6) for x, y in spec.positions.values()]
    placed: list[SynthOnion] = []

    def ok(cx, cy, r, ignore=None):
        if cx - r < 6 or cy - r < 6 or cx + r > spec.width_mm - 6 or cy + r > spec.height_mm - 6:
            return False
        for x0, y0, x1, y1 in keep_out:
            if x0 - r < cx < x1 + r and y0 - r < cy < y1 + r:
                return False
        for o in placed:
            if o is ignore:
                continue
            if math.hypot(o.cx_mm - cx, o.cy_mm - cy) < (o.d_mm + 2 * r) / 2 + 4:
                return False
        return True

    for k, c in enumerate(conds):
        d = float(rng.uniform(27, 40) if c == "small" else rng.uniform(46, 66))
        cond = "sound" if c == "small" else c
        o = SynthOnion(0, 0, d, float(rng.uniform(0.84, 0.97)), float(rng.uniform(0, math.pi)), cond)
        extra = 0.4 * d if cond == "sprouted" else 4
        for _ in range(400):
            if k < touching_pairs and placed and k % 2 == 1:
                prev = placed[-1]
                th = rng.uniform(0, 2 * math.pi)
                dist = (prev.d_mm * (0.5 + 0.5 * prev.aspect) / 2 + d * (0.5 + 0.5 * o.aspect) / 2) * 0.93
                cx, cy = prev.cx_mm + dist * math.cos(th), prev.cy_mm + dist * math.sin(th)
                if ok(cx, cy, d / 2 + extra, ignore=prev) and all(math.hypot(q.cx_mm - cx, q.cy_mm - cy) > (q.d_mm + d) / 2 + 4 for q in placed[:-1]):
                    o.cx_mm, o.cy_mm = cx, cy
                    break
            else:
                cx, cy = rng.uniform(10, spec.width_mm - 10), rng.uniform(10, spec.height_mm - 10)
                if ok(cx, cy, d / 2 + extra):
                    o.cx_mm, o.cy_mm = cx, cy
                    break
        else:
            continue
        placed.append(o)
    return placed


def _draw_onion(mat: np.ndarray, mask: np.ndarray, o: SynthOnion, ppm: float, rng) -> None:
    a = o.d_mm * ppm / 2
    b = a * o.aspect
    c = (o.cx_mm * ppm, o.cy_mm * ppm)
    ang = math.degrees(o.angle)
    H, W = mask.shape
    body = np.zeros((H, W), np.uint8)
    cv2.ellipse(body, (c, (2 * a, 2 * b), ang), 255, -1, cv2.LINE_AA)
    ca, sa = math.cos(o.angle), math.sin(o.angle)
    # neck (towards +major axis) and root tuft (towards -major axis): thin appendages
    neck = np.array([[a - 3, -b * 0.1], [a + 4 * ppm, 0], [a - 3, b * 0.1]])
    root = np.array([[-a + 2, -b * 0.12], [-a - 2.5 * ppm, -b * 0.05], [-a - 2.5 * ppm, b * 0.05], [-a + 2, b * 0.12]])
    rot = lambda P: np.stack([c[0] + P[:, 0] * ca - P[:, 1] * sa, c[1] + P[:, 0] * sa + P[:, 1] * ca], 1)
    # colour: red onion, radial shading (computed only in the onion's bounding window)
    x0, x1 = max(0, int(c[0] - a - 3)), min(W, int(c[0] + a + 4))
    y0, y1 = max(0, int(c[1] - a - 3)), min(H, int(c[1] + a + 4))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    u = ((xx - c[0]) * ca + (yy - c[1]) * sa) / a
    v = (-(xx - c[0]) * sa + (yy - c[1]) * ca) / b
    r2 = np.clip(u * u + v * v, 0, 1)
    hue = rng.uniform(-6, 6)
    base = np.array([95 + hue, 45, 150 + 2 * hue])            # BGR purple-red
    edge = np.array([60, 25, 95])
    col = base[None, None, :] * (1 - r2[..., None] * 0.6) + edge[None, None, :] * r2[..., None] * 0.6
    stripes = 0.08 * np.cos(v * 9 * math.pi)
    col = col * (1 + stripes[..., None])
    bw = body[y0:y1, x0:x1]
    sel = bw > 0
    alpha = (bw.astype(np.float32) / 255)[..., None]
    out = mat[y0:y1, x0:x1].astype(np.float32)
    out = out * (1 - alpha) + np.clip(col, 0, 255) * alpha
    if o.condition == "rotten":
        for _ in range(int(rng.integers(3, 6))):
            px, py = rng.uniform(-0.55, 0.55) * a, rng.uniform(-0.55, 0.55) * b
            rr = a * rng.uniform(0.22, 0.4)
            q = rot(np.array([[px, py]]))[0]
            dist = np.hypot(xx - q[0], yy - q[1])
            w = np.clip(1.25 - dist / rr, 0, 1) * sel
            out = out * (1 - w[..., None] * 0.93) + np.array([14, 20, 26]) * w[..., None] * 0.93
    if o.condition == "damaged":
        q = rot(np.array([[0.0, b * 0.55]]))[0]
        flesh = np.zeros((H, W), np.uint8)
        cv2.ellipse(flesh, ((q[0], q[1]), (0.7 * a, 0.55 * b), ang), 255, -1, cv2.LINE_AA)
        w = (flesh[y0:y1, x0:x1].astype(np.float32) / 255) * sel
        out = out * (1 - w[..., None]) + np.array([214, 206, 232]) * w[..., None]
    mat[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    cv2.fillPoly(mat, [rot(neck).astype(np.int32)], (60, 90, 140), cv2.LINE_AA)
    for t in np.linspace(-1, 1, 7):
        p0 = rot(np.array([[-a + 2, t * b * 0.1]]))[0]
        p1 = rot(np.array([[-a - 2.5 * ppm, t * b * 0.25]]))[0]
        cv2.line(mat, tuple(np.int32(p0)), tuple(np.int32(p1)), (140, 180, 200), 1, cv2.LINE_AA)
    full = body.copy()
    cv2.fillPoly(full, [rot(neck).astype(np.int32)], 255)
    if o.condition == "damaged":  # a cut notch removes a wedge of the outline
        notch = rot(np.array([[-a * 0.3, b + 3], [0, b * 0.35], [a * 0.3, b + 3]])).astype(np.int32)
        cv2.fillPoly(mat, [notch], PAPER, cv2.LINE_AA)
        cv2.fillPoly(full, [notch], 0)
    if o.condition == "sprouted":
        L, w = o.d_mm * ppm * 0.4, o.d_mm * ppm * 0.08
        shoot = rot(np.array([[a * 0.75, -w / 2], [a + L * 0.6, -w], [a + L, -w * 0.2], [a + L * 0.6, w * 0.4], [a * 0.75, w / 2]])).astype(np.int32)
        cv2.fillPoly(mat, [shoot], (58, 158, 94), cv2.LINE_AA)
        cv2.fillPoly(full, [shoot], 255)
    mask[full > 127] = 255


def make_photo(seed: int = 0, n: int = 9, touching_pairs: int = 2, small: int = 2,
               defects: dict | None = None, spec: MatSpec = DEFAULT_MAT, out_size=(1600, 1200),
               tilt: float = 0.12, blur: float | None = None, jpeg_quality: int | None = 88,
               render_ppm: float = 5.0):
    """Return (bgr_image, onions) with ground-truth polygons in photo pixels."""
    rng = np.random.default_rng(seed)
    defects = defects if defects is not None else {"sprouted": 1, "damaged": 1, "rotten": 2}
    mat = _render_mat(spec, render_ppm)
    onions = _place(rng, spec, n, touching_pairs, small, defects)
    # shadows first
    shadow = np.zeros(mat.shape[:2], np.float32)
    for o in onions:
        cv2.ellipse(shadow, ((o.cx_mm * render_ppm + 6, o.cy_mm * render_ppm + 8), (o.d_mm * render_ppm, o.d_mm * render_ppm * o.aspect), math.degrees(o.angle)), 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 7) * 0.22
    mat = (mat.astype(np.float32) * (1 - shadow[..., None])).astype(np.uint8)
    masks = []
    for o in onions:
        m = np.zeros(mat.shape[:2], np.uint8)
        _draw_onion(mat, m, o, render_ppm, rng)
        masks.append(m)
    # camera: map the mat into a tilted quadrilateral on a table
    W, H = out_size
    mw, mh = mat.shape[1], mat.shape[0]
    scale = min(W * 0.86 / mw, H * 0.86 / mh)
    cx, cy = W / 2 + rng.uniform(-20, 20), H / 2 + rng.uniform(-20, 20)
    hw, hh = mw * scale / 2, mh * scale / 2
    j = lambda: rng.uniform(-tilt, tilt)
    dst = np.array([[cx - hw * (1 + j()), cy - hh * (1 + j())], [cx + hw * (1 + j()), cy - hh * (1 + j())],
                    [cx + hw * (1 + j()), cy + hh * (1 + j())], [cx - hw * (1 + j()), cy + hh * (1 + j())]], np.float32)
    src = np.array([[0, 0], [mw, 0], [mw, mh], [0, mh]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    table = np.zeros((H, W, 3), np.uint8)
    table[:] = (70, 95, 120)
    photo = cv2.warpPerspective(mat, Hm, (W, H), dst=table, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_AREA)
    for o, m in zip(onions, masks):
        wm = cv2.warpPerspective(m, Hm, (W, H), flags=cv2.INTER_LINEAR)
        cnts, _ = cv2.findContours((wm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        o.polygon_img = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64) if cnts else None
    # light gradient, blur, noise, jpeg
    gx = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.95, 1.1), W)[None, :, None]
    gy = np.linspace(rng.uniform(0.9, 1.05), rng.uniform(0.9, 1.05), H)[:, None, None]
    photo = np.clip(photo.astype(np.float32) * gx * gy, 0, 255)
    sigma = rng.uniform(0.3, 1.1) if blur is None else blur
    if sigma > 0:
        photo = cv2.GaussianBlur(photo, (0, 0), sigma)
    photo = np.clip(photo + rng.normal(0, 3.0, photo.shape), 0, 255).astype(np.uint8)
    if jpeg_quality:
        ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        photo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return photo, [o for o in onions if o.polygon_img is not None]


def make_dataset(out_dir, n_photos: int = 12, seed: int = 0, **kw) -> dict:
    """Write synthetic photos + a COCO file (categories = conditions) + ruler CSV. Returns the COCO dict."""
    import csv
    import json
    from pathlib import Path
    from .coco import new_coco, add_image, add_annotation

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    coco = new_coco()
    rows = []
    for k in range(n_photos):
        img, onions = make_photo(seed=seed * 1000 + k, **kw)
        name = f"synthetic_{seed}_{k:03d}.jpg"
        cv2.imwrite(str(out / "images" / name), img)
        image_id = add_image(coco, name, img.shape[1], img.shape[0])
        for o in onions:
            add_annotation(coco, image_id, o.polygon_img, o.condition)
            c = o.polygon_img.mean(0)
            rows.append({"file_name": name, "x": round(float(c[0]), 1), "y": round(float(c[1]), 1), "ruler_mm": round(o.d_mm, 1)})
    (out / "annotations.json").write_text(json.dumps(coco))
    with open(out / "ruler.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "x", "y", "ruler_mm"])
        w.writeheader(); w.writerows(rows)
    return coco
