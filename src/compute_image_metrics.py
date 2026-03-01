"""Pre-compute blur, brightness, and contrast for BDD100K images."""

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


def _compute_one(img_path: Path, split: str) -> dict:
    result = {"image_name": img_path.name, "split": split}
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            **result,
            "blur_score": np.nan,
            "mean_brightness": np.nan,
            "contrast": np.nan,
        }
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        **result,
        "blur_score": round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2),
        "mean_brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
    }


def compute_all(workers: int | None = None) -> pd.DataFrame:
    workers = workers or min(os.cpu_count() or 4, 16)
    existing = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    done = set(existing["image_name"]) if len(existing) else set()

    todo = [
        (f, split)
        for split, d in IMAGE_DIRS.items()
        if d.exists()
        for f in sorted(d.iterdir())
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and f.name not in done
    ]
    if not todo:
        print(f"All {len(done)} images already processed.")
        return existing

    print(f"Processing {len(todo)} images ({len(done)} done) with {workers} threads...")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compute_one, p, s): p for p, s in todo}
        for f in tqdm(as_completed(futures), total=len(futures), unit="img"):
            results.append(f.result())

    combined = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(METRICS_PATH, index=False)
    print(f"Saved {len(combined)} rows to {METRICS_PATH}")
    return combined


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=None)
    compute_all(workers=p.parse_args().workers)
