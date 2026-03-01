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

# BDD100K images are 1280x720
IMG_WIDTH = 1280
IMG_HEIGHT = 720

VRU_CLASSES: frozenset[str] = frozenset({"person", "rider", "bike"})


def _resolve_splits(split: Literal["train", "val", "all"]) -> list[str]:
    """Return the list of split names for the given split selector."""
    return ["train", "val"] if split == "all" else [split]


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
    """Parse BDD100K labels into a DataFrame with one row per bounding box."""
    all_rows = []
    for s in _resolve_splits(split):
        all_rows.extend(_parse_single_file(LABEL_FILES[s], s))
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Drivable Area Parsing
# ---------------------------------------------------------------------------

DrivableAreaMap = dict[str, list[np.ndarray]]


def _parse_drivable_areas_single_file(path: Path) -> DrivableAreaMap:
    """Extract direct drivable area polygons from one label file."""
    data = json.loads(path.read_text())
    da_map: DrivableAreaMap = {}

    for image in data:
        img_name = image["name"]
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
            da_map[img_name] = polys

    return da_map


def parse_drivable_areas(
    split: Literal["train", "val", "all"] = "all",
) -> DrivableAreaMap:
    """Parse BDD100K direct drivable area polygons into {image_name: [polygon, ...]}."""
    combined: DrivableAreaMap = {}
    for s in _resolve_splits(split):
        combined.update(_parse_drivable_areas_single_file(LABEL_FILES[s]))
    return combined


def annotate_ego_lane(
    df: pd.DataFrame,
    da_map: DrivableAreaMap,
) -> pd.DataFrame:
    """Add 'in_ego_lane' column — True if VRU foot point is inside a drivable area."""
    from matplotlib.path import Path as MplPath

    result = df.copy()
    result["in_ego_lane"] = False

    vru_mask = result["category"].isin(VRU_CLASSES)
    if not vru_mask.any():
        return result

    vru_df = result.loc[vru_mask]

    for img_name, group in vru_df.groupby("image_name"):
        polys = da_map.get(img_name)
        if not polys:
            continue

        foot_points = np.column_stack([
            (group["x1"].values + group["x2"].values) / 2.0,
            group["y2"].values,
        ])

        in_any = np.zeros(len(group), dtype=bool)
        for verts in polys:
            path = MplPath(verts)
            in_any |= path.contains_points(foot_points)

        result.loc[group.index, "in_ego_lane"] = in_any

    return result
