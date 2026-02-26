"""Pre-compute image quality metrics for all BDD100K images.

Produces data/image_metrics.csv with one row per image. Supports incremental
runs — images already in the CSV are skipped. New metric columns can be added
by extending _compute_single_image(); existing rows keep their values and only
new images get the full set of columns.

Usage:
    python -m src.compute_image_metrics          # all images
    python -m src.compute_image_metrics --workers 8  # custom thread count
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.parser import IMAGE_DIRS

METRICS_PATH = Path(__file__).resolve().parent.parent / "data" / "image_metrics.csv"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _compute_single_image(img_path: Path, split: str) -> dict:
    """Compute quality metrics for one image.

    Returns a dict with image_name, split, and all metric columns.
    Add new metrics here — the CSV schema extends automatically.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            "image_name": img_path.name,
            "split": split,
            "blur_score": np.nan,
            "mean_brightness": np.nan,
            "contrast": np.nan,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = float(gray.mean())
    contrast = float(gray.std())

    return {
        "image_name": img_path.name,
        "split": split,
        "blur_score": round(blur_score, 2),
        "mean_brightness": round(mean_brightness, 2),
        "contrast": round(contrast, 2),
    }


def _collect_image_paths() -> list[tuple[Path, str]]:
    """Gather all (image_path, split) pairs from train and val directories."""
    pairs = []
    for split, directory in IMAGE_DIRS.items():
        if not directory.exists():
            continue
        for img_file in sorted(directory.iterdir()):
            if img_file.suffix.lower() in _IMAGE_EXTENSIONS:
                pairs.append((img_file, split))
    return pairs


def compute_all(workers: int | None = None) -> pd.DataFrame:
    """Compute metrics for all images, skipping those already in the CSV.

    Args:
        workers: Number of threads. Defaults to min(os.cpu_count(), 16).

    Returns:
        Complete DataFrame with metrics for all images.
    """
    if workers is None:
        workers = min(os.cpu_count() or 4, 16)

    # Load existing results for incremental support
    existing = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    already_done: set[str] = set(existing["image_name"]) if not existing.empty else set()

    all_pairs = _collect_image_paths()
    todo = [(p, s) for p, s in all_pairs if p.name not in already_done]

    if not todo:
        print(f"All {len(already_done)} images already processed.")
        return existing

    print(f"Processing {len(todo)} images ({len(already_done)} already done) with {workers} threads...")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute_single_image, p, s): p.name for p, s in todo}
        with tqdm(total=len(futures), desc="Computing image metrics", unit="img") as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    new_df = pd.DataFrame(results)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(METRICS_PATH, index=False)
    print(f"Saved {len(combined)} rows to {METRICS_PATH}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image quality metrics")
    parser.add_argument("--workers", type=int, default=None, help="Thread count")
    args = parser.parse_args()
    compute_all(workers=args.workers)
