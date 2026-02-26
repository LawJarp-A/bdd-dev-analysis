# BDD100K Object Detection — Data Analysis

Exploratory data analysis of the [BDD100K](https://www.vis.xyz/bdd100k/) driving dataset, focused on the 2D object detection task. The project parses raw JSON labels into a structured DataFrame, computes distribution and anomaly statistics, and serves an interactive Streamlit dashboard for visual exploration.

Built as part of a Bosch Applied Computer Vision interview assignment.

## Quick Start (Docker)

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). The dataset is downloaded automatically on first run if the `data/` directory is empty.

## Local Development Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.

```bash
# Install dependencies
uv sync

# Download dataset (auto-skips if already present)
uv run python -m src.download_data

# Launch dashboard
uv run streamlit run src/dashboard.py
```

## Image Quality Metrics (optional)

Pre-compute blur, brightness, and contrast metrics to enable the full Safety-Critical Edge Cases analysis:

```bash
uv run python -m src.compute_image_metrics          # uses all available cores
uv run python -m src.compute_image_metrics --workers 8  # custom thread count
```

This produces `data/image_metrics.csv` (~80K rows). The script supports incremental runs and shows a tqdm progress bar.

## Dashboard

The dashboard provides five tabs:

- **Overview** — Total annotations, unique images, class distribution chart, co-occurrence matrix, train/val split balance (chi-squared test), and attribute summary tables.
- **Class Deep Dive** — Objects-per-image distribution, occlusion/truncation rates, per-class bounding box area distribution, and spatial heatmaps.
- **Anomalies** — Per-class size outliers, extreme size boxes (tiny and huge), and extreme aspect ratios. Each anomaly includes an explanation and a sample image viewer with anomalous bboxes highlighted in red (other boxes shown in faded gray).
- **Safety-Critical Edge Cases** — Scenarios where detection failures have the highest real-world cost: tiny VRUs at night/rain, occluded pedestrians near cars, crowded night intersections, truncated pedestrians, blurry/dark frames with VRUs, rare condition combos, and a blind spots analysis of unlabeled object types.
- **Sample Browser** — Browse images with color-coded bounding box overlays. Modes: Most Crowded, Rare Classes, Single-Class Images, Per-Class Outliers, Highly Occluded, Random.

## Project Structure

```
bdd-dev-analysis/
├── data/                      # Dataset (gitignored, auto-downloaded)
├── docs/                      # Analysis report
├── src/
│   ├── __init__.py
│   ├── compute_image_metrics.py  # Pre-compute blur/brightness/contrast
│   ├── download_data.py       # Google Drive download + extraction
│   ├── parser.py              # JSON label parsing to DataFrame
│   ├── analysis.py            # Statistics, anomaly detection, safety queries, plots
│   └── dashboard.py           # Streamlit application
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependency versions
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Dataset

**BDD100K** — 80,000 images (70K train / 10K val) at 1280x720 resolution, with 2D bounding box annotations across 10 object classes:

| Class | Class | Class |
|-------|-------|-------|
| bike | bus | car |
| motor | person | rider |
| traffic light | traffic sign | train |
| truck | | |

Each annotation includes bounding box coordinates, occlusion/truncation flags, and image-level attributes (weather, scene, time of day).

The dataset is hosted on Google Drive and downloaded automatically by `src/download_data.py` when the `data/` directory is missing.
