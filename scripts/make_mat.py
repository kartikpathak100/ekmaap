"""Print the grading mat: A4 landscape with 4 ArUco markers, plus a loose 50 mm size card.

    python scripts/make_mat.py --out ekmaap_mat.pdf

Print at 100% ("actual size"), then check the 100 mm bar with a ruler. The markers are what give
scale AND tilt correction, so do not fold, crop or cover them.
"""
import argparse
import io

import cv2
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from _common import *  # noqa: F401,F403  (sets sys.path)
from ekmaap.calibration import ARUCO_DICT, DEFAULT_MAT


def marker_png(marker_id: int, px: int = 600) -> ImageReader:
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    img = cv2.aruco.generateImageMarker(d, marker_id, px)
    ok, buf = cv2.imencode(".png", img)
    return ImageReader(io.BytesIO(buf.tobytes()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ekmaap_mat.pdf")
    args = ap.parse_args()
    spec = DEFAULT_MAT
    W, H = landscape(A4)
    c = canvas.Canvas(args.out, pagesize=(W, H))
    for mid, (x, y) in spec.positions.items():
        # PDF origin is bottom-left; mat coordinates are top-left
        c.drawImage(marker_png(mid), x * mm, H - (y + spec.marker_mm) * mm, spec.marker_mm * mm, spec.marker_mm * mm)
    c.setFont("Helvetica", 9)
    c.drawCentredString(W / 2, H - 14 * mm, "EkMaap grading mat - print at 100% / actual size. Onions go between the markers, one layer, not touching the markers.")
    c.setLineWidth(0.8)
    c.line(W / 2 - 50 * mm, 20 * mm, W / 2 + 50 * mm, 20 * mm)
    for t in range(0, 101, 10):
        c.line(W / 2 - 50 * mm + t * mm, 20 * mm, W / 2 - 50 * mm + t * mm, (22 if t % 50 else 24) * mm)
    c.drawCentredString(W / 2, 14 * mm, "This bar must measure exactly 100 mm. If not, fix the printer scaling and print again.")
    c.showPage()
    c.setPageSize(A4)
    W2, H2 = A4
    s = spec.card_mm
    c.drawImage(marker_png(spec.card_id), (W2 - s * mm) / 2, H2 - 40 * mm - s * mm, s * mm, s * mm)
    c.setFont("Helvetica", 10)
    c.drawCentredString(W2 / 2, H2 - 50 * mm - s * mm, f"Loose size card (id {spec.card_id}, {s:.0f} mm). Use only when the mat is not available: tilt cannot be fully corrected.")
    c.save()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
