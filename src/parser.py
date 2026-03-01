"""Parse BDD100K JSON labels into a flat pandas DataFrame."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
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

IMG_WIDTH = 1280
IMG_HEIGHT = 720

VRU_CLASSES: frozenset[str] = frozenset({"person", "rider", "bike"})


def _parse_single_file(path: Path, split: str) -> list[dict]:
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
            w, h = x2 - x1, y2 - y1
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
    """Parse BDD100K labels into a DataFrame with one row per bounding box."""
    splits = ["train", "val"] if split == "all" else [split]
    all_rows = []
    for s in splits:
        all_rows.extend(_parse_single_file(LABEL_FILES[s], s))
    return pd.DataFrame(all_rows)


def _parse_drivable_single(path: Path) -> dict[str, list[np.ndarray]]:
    data = json.loads(path.read_text())
    da_map: dict[str, list[np.ndarray]] = {}
    for image in data:
        polys: list[np.ndarray] = []
        for label in image.get("labels", []):
            if label.get("category") != "drivable area":
                continue
            if label.get("attributes", {}).get("areaType") != "direct":
                continue
            for poly2d in label.get("poly2d", []):
                vertices = poly2d.get("vertices", [])
                if len(vertices) >= 3:
                    polys.append(np.array(vertices, dtype=np.float64))
        if polys:
            da_map[image["name"]] = polys
    return da_map


def parse_drivable_areas(
    split: Literal["train", "val", "all"] = "all",
) -> dict[str, list[np.ndarray]]:
    """Parse BDD100K direct drivable area polygons."""
    combined: dict[str, list[np.ndarray]] = {}
    for s in (["train", "val"] if split == "all" else [split]):
        combined.update(_parse_drivable_single(LABEL_FILES[s]))
    return combined


def annotate_ego_lane(
    df: pd.DataFrame,
    da_map: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    """Add 'in_ego_lane' column — True if VRU foot point is inside a drivable area."""
    from matplotlib.path import Path as MplPath

    result = df.copy()
    result["in_ego_lane"] = False

    vru_mask = result["category"].isin(VRU_CLASSES)
    if not vru_mask.any():
        return result

    for img_name, group in result.loc[vru_mask].groupby("image_name"):
        polys = da_map.get(img_name)
        if not polys:
            continue
        foot_points = np.column_stack(
            [
                (group["x1"].values + group["x2"].values) / 2.0,
                group["y2"].values,
            ]
        )
        in_any = np.zeros(len(group), dtype=bool)
        for verts in polys:
            in_any |= MplPath(verts).contains_points(foot_points)
        result.loc[group.index, "in_ego_lane"] = in_any

    return result
