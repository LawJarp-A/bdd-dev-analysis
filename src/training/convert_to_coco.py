"""Convert BDD100K annotations to COCO format for training."""

import json
import os
from pathlib import Path

from tqdm import tqdm

from src.parser import DETECTION_CLASSES, IMAGE_DIRS, IMG_HEIGHT, IMG_WIDTH, LABEL_FILES

CLASS_TO_ID = {cls: i for i, cls in enumerate(DETECTION_CLASSES)}
COCO_BASE = Path(__file__).resolve().parent.parent.parent / "data" / "bdd100k_coco"
SPLIT_MAP = {"train": "train", "val": "valid"}  # RF-DETR uses "valid" not "val"


def convert_split(split: str) -> dict:
    coco_split = SPLIT_MAP[split]
    out_dir = COCO_BASE / coco_split
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_images, coco_annotations = [], []
    coco_categories = [
        {"id": i, "name": cls, "supercategory": cls}
        for i, cls in enumerate(DETECTION_CLASSES)
    ]

    ann_id = 0
    stats = {cls: 0 for cls in DETECTION_CLASSES}
    raw = json.loads(LABEL_FILES[split].read_text())

    for img_id, entry in enumerate(tqdm(raw, desc=f"Converting {split}")):
        img_name = entry["name"]
        coco_images.append(
            {
                "id": img_id,
                "file_name": img_name,
                "width": IMG_WIDTH,
                "height": IMG_HEIGHT,
            }
        )

        src = (IMAGE_DIRS[split] / img_name).resolve()
        dst = out_dir / img_name
        if not dst.exists() and src.exists():
            os.symlink(src, dst)

        for label in entry.get("labels", []):
            if "box2d" not in label:
                continue
            cat = label["category"]
            if cat not in CLASS_TO_ID:
                continue

            b = label["box2d"]
            x1 = max(0.0, min(float(b["x1"]), IMG_WIDTH))
            y1 = max(0.0, min(float(b["y1"]), IMG_HEIGHT))
            x2 = max(0.0, min(float(b["x2"]), IMG_WIDTH))
            y2 = max(0.0, min(float(b["y2"]), IMG_HEIGHT))
            if x2 <= x1 or y2 <= y1:
                continue

            w, h = x2 - x1, y2 - y1
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CLASS_TO_ID[cat],
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            stats[cat] += 1

    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }
    json_path = out_dir / "_annotations.coco.json"
    json_path.write_text(json.dumps(coco_dict))
    print(f"  Wrote {json_path} ({len(coco_images)} images, {ann_id} annotations)")
    return {
        "n_images": len(coco_images),
        "n_annotations": ann_id,
        "class_counts": stats,
    }


def main() -> None:
    total_anns = 0
    for split in ("train", "val"):
        print(f"\nConverting {split} split...")
        stats = convert_split(split)
        total_anns += stats["n_annotations"]
        print(f"  {stats['n_images']} images, {stats['n_annotations']} annotations")
        for cls, count in stats["class_counts"].items():
            print(f"    {cls}: {count:,}")
    print(f"\nTotal annotations: {total_anns:,}")
    print(f"Output directory: {COCO_BASE}")


if __name__ == "__main__":
    main()
