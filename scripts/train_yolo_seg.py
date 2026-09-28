"""OPTIONAL GPU model: fine-tune YOLO-seg on the rectified dataset from coco_to_yolo.py.

    pip install ultralytics            # pulls PyTorch; use a GPU (Colab / Kaggle free tiers work)
    python scripts/train_yolo_seg.py --data data/yolo/data.yaml --out models/yolo

Not exercised in this repository's tests (PyTorch is not part of the test environment); treat the first run as a trial.
Licence: Ultralytics is AGPL-3.0 - fine for an open hackathon repo, but a government deployment needs
either AGPL compliance or an Ultralytics enterprise licence. The pixel-forest segmenter has no such issue.
After training, register models/yolo/weights/best.pt as a segmenter (it loads through
ekmaap.segmentation.YoloSegmenter) and compare it with scripts/evaluate.py on the test split.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="models/yolo")
ap.add_argument("--model", default="yolo11n-seg.pt", help="pretrained start point (downloaded by ultralytics)")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=896, help="rectified mat is 891 x 630 px at 3 px/mm")
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

model = YOLO(a.model)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=a.out, name="run", exist_ok=True,
            patience=25, degrees=180, fliplr=0.5, flipud=0.5, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, scale=0.1, mosaic=0.5)
metrics = model.val(data=a.data, split="test")
print("test mask mAP50-95:", metrics.seg.map, " mAP50:", metrics.seg.map50)
