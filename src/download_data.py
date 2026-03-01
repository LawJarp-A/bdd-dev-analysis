"""Download BDD100K dataset from Google Drive."""

import zipfile

import gdown

from src.compute_image_metrics import METRICS_PATH
from src.parser import DATA_DIR

GDRIVE_URL = "https://drive.google.com/uc?id=1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT"
METRICS_GDRIVE_URL: str | None = None
ZIP_PATH = DATA_DIR / "bdd100k_data.zip"

_EXPECTED = [
    DATA_DIR / "bdd100k_labels_release",
    DATA_DIR / "bdd100k_images_100k",
]


def data_exists() -> bool:
    return all(d.is_dir() for d in _EXPECTED)


def download_and_extract() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print(f"Zip already exists at {ZIP_PATH}, skipping download.")
    else:
        print(f"Downloading BDD100K dataset to {ZIP_PATH}...")
        gdown.download(GDRIVE_URL, str(ZIP_PATH), quiet=False)

    print(f"Extracting to {DATA_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Extraction complete.")


def download_metrics() -> None:
    if METRICS_PATH.exists():
        print("Image metrics already present, skipping download.")
        return
    if METRICS_GDRIVE_URL is None:
        print(
            "No metrics URL configured. Run `python -m src.compute_image_metrics` to generate locally."
        )
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading image metrics to {METRICS_PATH}...")
    gdown.download(METRICS_GDRIVE_URL, str(METRICS_PATH), quiet=False)


def ensure_data() -> None:
    if data_exists():
        print("Dataset already present, skipping download.")
    else:
        download_and_extract()
        if not data_exists():
            missing = [str(d) for d in _EXPECTED if not d.is_dir()]
            raise RuntimeError(f"Download done but directories missing: {missing}")
    download_metrics()


if __name__ == "__main__":
    ensure_data()
