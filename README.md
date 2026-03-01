# BDD100K Object Detection Pipeline

End-to-end object detection on [BDD100K](https://www.vis.xyz/bdd100k/): data analysis, model training (RF-DETR), and evaluation — all in an interactive Streamlit dashboard.

## Quick Start

**Docker (recommended):**
```bash
docker build -t bdd-analysis .
docker run -p 8501:8501 -v ./data:/app/data bdd-analysis
```

**Local:**
```bash
uv sync
uv run python -m src.download_data
uv run streamlit run src/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501). Data downloads automatically on first run.

## Project Phases

### Phase 1 — Data Analysis

Parses BDD100K JSON labels into structured DataFrames and provides interactive exploration across 5 dashboard tabs:

- **Overview** — Class distribution, co-occurrence matrix, train/val split balance
- **Class Deep Dive** — Per-class size distributions, occlusion rates, spatial heatmaps
- **Anomalies** — Size outliers, extreme aspect ratios with sample image viewer
- **Safety-Critical Edge Cases** — Tiny pedestrians at night, occluded VRUs, crowded intersections, driving-lane filtering
- **Sample Browser** — Browse images with color-coded bounding boxes (most crowded, rare classes, outliers, etc.)

### Phase 2 — Model Training

**Model:** [RF-DETR Large](https://github.com/roboflow/rf-detr) (33.6M params) — a real-time DETR variant with DINOv2 backbone.

**Custom dataloader** (`src/training/dataset.py`) loads BDD100K images with albumentations augmentations and feeds them directly into RF-DETR's model. The training loop (`src/training/train.py`) replicates RF-DETR's recipe:

- Layer-wise LR decay for backbone, cosine schedule with warmup
- EMA, AMP (bfloat16), gradient accumulation, gradient clipping
- Checkpoint saving (periodic + best)

```bash
# Train on full dataset
uv run python -m src.training.train --epochs 50 --batch-size 16

# Quick test (1 epoch, 1% of data)
uv run python -m src.training.train --epochs 1 --fraction 0.01 --batch-size 2
```

### Phase 3 — Evaluation

Evaluates the trained model on the validation set using COCO metrics. Dashboard tab 6 shows:

- **Quantitative** — mAP@50/75, per-class AP, precision-recall curves, confusion matrix
- **Failure Analysis** — Failure clustering by weather/time/object size, correlation analysis
- **Qualitative** — Side-by-side GT vs predictions browser

```bash
# Run inference on val set
uv run python -m src.evaluation.run_inference

# Compute and cache eval metrics
uv run python -m src.evaluation.metrics
```

## Project Structure

```
src/
├── parser.py                  # JSON label parsing
├── analysis.py                # Statistics, anomaly detection, safety queries
├── compute_image_metrics.py   # Blur/brightness/contrast metrics
├── download_data.py           # Dataset download
├── dashboard.py               # Streamlit app (6 tabs)
├── training/
│   ├── dataset.py             # Custom BDD100K PyTorch Dataset
│   ├── train.py               # Training loop with RF-DETR internals
│   └── convert_to_coco.py     # BDD100K → COCO format conversion
└── evaluation/
    ├── run_inference.py        # Val set inference
    └── metrics.py              # COCO eval, confusion matrix, failure analysis
```

## Dataset

**BDD100K** — 80K images (70K train / 10K val), 1280x720, 10 detection classes: bike, bus, car, motor, person, rider, traffic light, traffic sign, train, truck.

Each annotation includes bounding box, occlusion/truncation flags, and image-level attributes (weather, scene, time of day).
