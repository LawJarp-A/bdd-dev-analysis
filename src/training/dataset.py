"""BDD100K object detection dataset with albumentations transforms.

Outputs targets in normalised cxcywh format for RF-DETR / DETR models.
"""

import json
import random
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from rfdetr.util.misc import NestedTensor

from src.parser import (
    DETECTION_CLASSES,
    IMAGE_DIRS,
    IMG_HEIGHT,
    IMG_WIDTH,
    LABEL_FILES,
)

CLASS_TO_ID = {cls: i for i, cls in enumerate(DETECTION_CLASSES)}


def get_train_transforms(img_size: int = 640) -> A.Compose:
    """Training augmentations with bbox-safe transforms."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.2,
        ),
    )


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """Validation transforms: resize + normalize."""
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.2,
        ),
    )


def _xyxy_to_cxcywh_normalised(boxes: torch.Tensor, img_size: int) -> torch.Tensor:
    """Convert [x1,y1,x2,y2] pixel coords to normalised [cx,cy,w,h] in [0,1]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2.0 / img_size
    cy = (y1 + y2) / 2.0 / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return torch.stack([cx, cy, w, h], dim=-1)


class BDD100KDataset(Dataset):
    """Returns (image_tensor, target) with boxes in normalised cxcywh format."""

    def __init__(
        self,
        split: str = "train",
        transforms: Optional[A.Compose] = None,
        img_size: int = 640,
        fraction: float = 1.0,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.split = split
        self.img_size = img_size
        self.image_dir = IMAGE_DIRS[split]
        self.transforms = transforms or (
            get_train_transforms(img_size) if split == "train"
            else get_val_transforms(img_size)
        )
        self._images, self._annotations = self._load_labels(LABEL_FILES[split])

        if 0.0 < fraction < 1.0:
            n_keep = max(1, int(len(self._images) * fraction))
            indices = sorted(random.sample(range(len(self._images)), n_keep))
            self._images = [self._images[i] for i in indices]
            self._annotations = [self._annotations[i] for i in indices]

    @staticmethod
    def _load_labels(label_path: Path) -> tuple[list[dict], list[list[dict]]]:
        raw = json.loads(label_path.read_text())
        images, annotations = [], []

        for entry in raw:
            boxes = []
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
                boxes.append({"bbox": [x1, y1, x2, y2], "class_id": CLASS_TO_ID[cat]})
            images.append({"file_name": entry["name"]})
            annotations.append(boxes)

        return images, annotations

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_info = self._images[idx]
        anns = self._annotations[idx]

        img_path = self.image_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = [a["bbox"] for a in anns]
        class_labels = [a["class_id"] for a in anns]

        transformed = self.transforms(
            image=image, bboxes=bboxes, class_labels=class_labels
        )

        image_tensor = transformed["image"]
        out_boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32).reshape(-1, 4)

        if out_boxes.numel() > 0:
            out_boxes = _xyxy_to_cxcywh_normalised(out_boxes, self.img_size)

        return image_tensor, {
            "boxes": out_boxes,
            "labels": torch.as_tensor(transformed["class_labels"], dtype=torch.int64),
            "image_id": torch.tensor(idx),
        }


def collate_fn(
    batch: list[tuple[torch.Tensor, dict]],
) -> tuple[NestedTensor, list[dict]]:
    """Collate images into a NestedTensor (no padding needed, all same size)."""
    images, targets = zip(*batch)
    stacked = torch.stack(images)
    mask = torch.zeros(stacked.shape[0], stacked.shape[2], stacked.shape[3], dtype=torch.bool)
    return NestedTensor(stacked, mask), list(targets)


if __name__ == "__main__":
    ds = BDD100KDataset(split="train", fraction=0.001)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    samples, targets = next(iter(loader))
    print(f"Images: {samples.tensors.shape}, mask: {samples.mask.shape}")
    for i, t in enumerate(targets):
        print(f"  Image {i}: {t['boxes'].shape[0]} boxes (cxcywh), labels={t['labels'].tolist()}")
