"""Download BDD100K dataset and pre-computed metrics from Google Drive."""

import zipfile

import gdown

from src.compute_image_metrics import METRICS_PATH
from src.parser import DATA_DIR

GDRIVE_URL = "https://drive.google.com/uc?id=1NgWX5YfEKbloAKX9l8kUVJFpWFlUO8UT"
# Set this to the actual Google Drive file ID once the metrics CSV is uploaded
METRICS_GDRIVE_URL: str | None = None

ZIP_PATH = DATA_DIR / "bdd100k_data.zip"

EXPECTED_DIRS = [
    DATA_DIR / "bdd100k_labels_release",
    DATA_DIR / "bdd100k_images_100k",
]


def data_exists() -> bool:
    """Check if both labels and images directories exist."""
    return all(d.is_dir() for d in EXPECTED_DIRS)


def download_and_extract() -> None:
    """Download dataset zip from Google Drive and extract to data/."""
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
    """Download pre-computed image metrics CSV if not present."""
    if METRICS_PATH.exists():
        print("Image metrics already present, skipping download.")
        return

    if METRICS_GDRIVE_URL is None:
        print(
            "No metrics download URL configured. "
            "Run `python -m src.compute_image_metrics` to generate locally."
        )
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading image metrics to {METRICS_PATH}...")
    gdown.download(METRICS_GDRIVE_URL, str(METRICS_PATH), quiet=False)


def ensure_data() -> None:
    """Download and extract data if it doesn't already exist."""
    if data_exists():
        print("Dataset already present, skipping download.")
    else:
        download_and_extract()
        if not data_exists():
            missing = [str(d) for d in EXPECTED_DIRS if not d.is_dir()]
            raise RuntimeError(
                f"Download completed but expected directories not found: {missing}"
            )

    download_metrics()


if __name__ == "__main__":
    ensure_data()
