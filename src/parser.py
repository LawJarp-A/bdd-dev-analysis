"""Parse BDD100K JSON labels into a flat pandas DataFrame."""

import json
from pathlib import Path
from typing import Literal

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_LABELS_BASE = DATA_DIR / "bdd100k_labels_release" / "bdd100k" / "labels"
_IMAGES_BASE = DATA_DIR / "bdd100k_images_100k" / "bdd100k" / "images" / "100k"

LABEL_FILES = {
    "train": _LABELS_BASE / "bdd100k_labels_images_train.json",
    "val": _LABELS_BASE / "bdd100k_labels_images_val.json",
}

IMAGE_DIRS = {
    "train": _IMAGES_BASE / "train",
    "val": _IMAGES_BASE / "val",
}

DETECTION_CLASSES = [
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
]

# BDD100K images are 1280x720
IMG_WIDTH = 1280
IMG_HEIGHT = 720


def _parse_single_file(path: Path, split: str) -> list[dict]:
    """Parse one JSON label file into a list of row dicts."""
    data = json.loads(path.read_text())

    rows = []
    for image in data:
        img_name = image["name"]
        attrs = image.get("attributes", {})
        weather = attrs.get("weather", "unknown")
        scene = attrs.get("scene", "unknown")
        timeofday = attrs.get("timeofday", "unknown")

        for label in image.get("labels", []):
            if "box2d" not in label:
                continue
            category = label["category"]
            if category not in DETECTION_CLASSES:
                continue

            box = label["box2d"]
            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]
            w = x2 - x1
            h = y2 - y1

            label_attrs = label.get("attributes", {})
            rows.append(
                {
                    "image_name": img_name,
                    "category": category,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": w,
                    "height": h,
                    "area": w * h,
                    "aspect_ratio": w / h if h > 0 else 0.0,
                    "occluded": label_attrs.get("occluded", False),
                    "truncated": label_attrs.get("truncated", False),
                    "weather": weather,
                    "scene": scene,
                    "timeofday": timeofday,
                    "split": split,
                }
            )
    return rows


def parse_labels(split: Literal["train", "val", "all"] = "all") -> pd.DataFrame:
    """Parse BDD100K labels into a DataFrame.

    Args:
        split: Which split to load -- "train", "val", or "all".

    Returns:
        DataFrame with one row per bounding box annotation.
    """
    splits = ["train", "val"] if split == "all" else [split]
    all_rows = []
    for s in splits:
        all_rows.extend(_parse_single_file(LABEL_FILES[s], s))
    return pd.DataFrame(all_rows)
