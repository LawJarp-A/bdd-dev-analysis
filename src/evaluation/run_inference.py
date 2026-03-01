"""Run RF-DETR inference on BDD100K validation set."""

import csv
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.parser import DETECTION_CLASSES

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
COCO_ANN = DATA_DIR / "bdd100k_coco" / "valid" / "_annotations.coco.json"
VAL_IMG_DIR = DATA_DIR / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / "val"
PRED_JSON = DATA_DIR / "predictions" / "val_predictions.json"
DEFAULT_CKPT = Path(__file__).resolve().parent.parent.parent / "runs" / "rfdetr_bdd100k_5ep" / "checkpoint_best_ema.pth"


def main():
    from rfdetr import RFDETRLarge

    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CKPT)
    model = RFDETRLarge(pretrain_weights=ckpt, num_classes=11)
    model.model.class_names = DETECTION_CLASSES

    with open(COCO_ANN) as f:
        coco = json.load(f)
    fname_to_id = {img["file_name"]: img["id"] for img in coco["images"]}

    results = []
    for fname, img_id in tqdm(fname_to_id.items(), desc="Inference"):
        img_path = VAL_IMG_DIR / fname
        if not img_path.exists():
            continue
        dets = model.predict(Image.open(img_path).convert("RGB"), threshold=0.01)
        for xyxy, cls_id, conf in zip(dets.xyxy, dets.class_id, dets.confidence):
            x1, y1, x2, y2 = xyxy.tolist()
            results.append({
                "image_id": img_id, "category_id": int(cls_id),
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(conf), 4),
            })

    PRED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_JSON, "w") as f:
        json.dump(results, f)
    with open(PRED_JSON.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "category_id", "x", "y", "w", "h", "score"])
        for r in results:
            w.writerow([r["image_id"], r["category_id"], *r["bbox"], r["score"]])
    print(f"Saved {len(results)} detections from {len(fname_to_id)} images")


if __name__ == "__main__":
    main()
