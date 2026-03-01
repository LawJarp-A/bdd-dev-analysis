# BDD100K Object Detection Pipeline

End-to-end object detection on [BDD100K](https://www.vis.xyz/bdd100k/): data analysis, model training (RF-DETR), and evaluation — all in an interactive Streamlit dashboard.

## Quick Start

**Docker (recommended):**
```bash
docker compose up          # builds and runs everything

# or manually:
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

**Goal:** Understand the BDD100K dataset thoroughly before training — identify class imbalances, annotation quality issues, and edge cases that could affect model performance.

**What we did and why:**

1. **Class distribution analysis** — BDD100K is heavily imbalanced: `car` dominates with ~720K training annotations while `train` has only ~170. Understanding this upfront tells us which classes the model will likely struggle with and whether class-balanced sampling or loss weighting is needed.

2. **Train/val split balance** — A chi-squared test confirms the class proportions are consistent across splits, meaning validation performance is a fair estimate of training performance (no distribution shift between splits).

3. **Co-occurrence matrix** — Shows which classes frequently appear together. Cars co-occur with nearly everything, while rare classes (train, motor) appear in isolation. This matters because co-occurring classes provide contextual cues during detection — a lone `rider` without nearby `bike` or `car` is harder to classify.

4. **Bounding box size distributions** — Per-class area analysis reveals that traffic lights and signs are predominantly small objects (<1% of image area), while buses and trucks are large. This directly impacts which anchor scales or feature pyramid levels the model needs to handle well.

5. **Occlusion and truncation rates** — Riders (55% occluded) and bikes (47% occluded) are the most frequently occluded classes. Heavily occluded objects are harder to detect because visible features are incomplete. This analysis predicts which classes will have lower recall.

6. **Anomaly detection** — Three types of per-class anomalies are identified:
   - *Double-degraded* boxes (both occluded AND truncated) — the hardest cases for any detector
   - *Crowding anomalies* — images where a single class appears far more than typical (e.g., 30+ cars in one frame)
   - *Extreme aspect ratios* — unusually shaped boxes per class that may indicate annotation errors or unusual viewpoints

7. **Safety-critical edge cases** — Specifically targets autonomous driving failure modes: tiny pedestrians at night/rain, occluded VRUs near cars, crowded night intersections, and truncated persons at image edges. These are the scenarios where detection failures have the highest real-world cost.

8. **Image quality metrics** — Computes blur (Laplacian variance), mean brightness, and contrast for every validation image. These are later correlated with model recall to quantify how image quality affects detection.

The dashboard presents all of this across 5 interactive tabs (Overview, Class Deep Dive, Anomalies, Safety-Critical Edge Cases, Sample Browser).

### Phase 2 — Model Selection: Why RF-DETR

**Model:** [RF-DETR Large](https://github.com/roboflow/rf-detr) (33.6M params)

**Why RF-DETR over other detectors:**

- **DETR-family advantages** — RF-DETR is a Detection Transformer, meaning it uses set-based prediction with bipartite matching instead of anchor boxes and NMS. This eliminates hand-tuned anchor design and NMS thresholds, which is especially valuable for BDD100K where object scales vary dramatically (tiny traffic lights to large trucks in the same frame).

- **DINOv2 backbone** — RF-DETR uses a DINOv2 vision transformer backbone pretrained with self-supervised learning on 142M images. DINOv2 features are more robust to domain shifts (weather, lighting) compared to ImageNet-supervised backbones like ResNet. This matters for BDD100K where conditions range from clear daytime to rainy night.

- **Multi-scale feature handling** — The architecture includes a feature pyramid that processes objects at multiple scales natively. Given that BDD100K has extreme scale variation (AP_small and AP_large differ by 4-5x across classes), strong multi-scale handling is critical.

- **Real-time capable** — RF-DETR achieves competitive accuracy while maintaining real-time inference speeds, which aligns with the autonomous driving use case where latency matters.

- **Modern training recipe** — Layer-wise LR decay, cosine schedule with warmup, EMA, AMP (bfloat16), and gradient clipping are all built into the training pipeline. These techniques collectively improve convergence stability and final accuracy.

**Alternatives considered:**
- *YOLOv8/YOLOv11* — Strong real-time detectors but rely on anchor-based detection with NMS, requiring per-dataset tuning of anchor scales. Less elegant for BDD100K's extreme scale range.
- *Co-DETR / DINO* — Higher accuracy on COCO benchmarks but significantly larger and slower. Overkill for this evaluation scope.
- *Faster R-CNN* — Well-understood baseline but two-stage pipeline is slower and the region proposal network adds complexity without clear benefit over modern one-stage detectors.

**RF-DETR Results on BDD100K validation set:**

| Metric | Value |
|--------|-------|
| mAP@50 | 59.6% |
| mAP@50:95 | 33.8% |
| mAP@75 | 32.2% |
| AP (small objects) | 13.4% |
| AP (medium objects) | 38.6% |
| AP (large objects) | 61.6% |
| AR@500 | 47.3% |

**Per-class AP@50:**

| Class | AP@50 | GT Count |
|-------|-------|----------|
| car | 81.4% | 102,506 |
| traffic sign | 73.9% | 34,908 |
| traffic light | 68.7% | 26,885 |
| bus | 68.5% | 1,597 |
| truck | 68.5% | 4,245 |
| person | 66.4% | 13,262 |
| motor | 56.6% | 452 |
| bike | 54.2% | 1,007 |
| rider | 51.3% | 649 |
| train | 6.3% | 15 |

The model performs well on frequent, large classes (car: 81.4% AP@50) and struggles with rare classes (train: 6.3% AP@50 with only 15 validation samples) and small objects (AP_small = 13.4% vs AP_large = 61.6%).

**Custom training pipeline** (`src/training/dataset.py`, `src/training/train.py`) loads BDD100K images with albumentations augmentations and replicates RF-DETR's training recipe:

```bash
# Train on full dataset
uv run python -m src.training.train --epochs 50 --batch-size 16

# Quick test (1 epoch, 1% of data)
uv run python -m src.training.train --epochs 1 --fraction 0.01 --batch-size 2
```

### Phase 3 — Evaluation

**Goal:** Go beyond aggregate metrics to understand *where* and *why* the model fails, and connect those failures back to the data properties identified in Phase 1.

**Metrics chosen and why:**

1. **COCO mAP@50 and mAP@50:95** — The standard object detection metrics. mAP@50 measures detection at a lenient IoU threshold (good for "did we find the object?"), while mAP@50:95 averages over IoU thresholds from 0.5 to 0.95 (penalizes imprecise localization). We report both because autonomous driving needs both detection *and* precise localization.

2. **AP by object size (small/medium/large)** — BDD100K has extreme scale variation. Reporting AP separately by size reveals that the model's overall mAP is dragged down by small objects (13.4%) while large objects are well-detected (61.6%). This directly maps to the size distribution analysis from Phase 1.

3. **Per-class AP** — Essential given BDD100K's class imbalance. A single mAP number hides that `car` (81.4%) and `train` (6.3%) have wildly different performance. Per-class AP lets us correlate performance with training sample count, occlusion rate, and typical object size from Phase 1.

4. **Precision-recall curves** — Show the full tradeoff between precision and recall at every confidence threshold, per class. More informative than a single AP number — they reveal whether a class has a sharp or gradual precision dropoff as recall increases.

5. **Confusion matrix** — Shows which classes get confused with each other (e.g., bike vs motor, car vs truck) and which go undetected (confused with "background"). This identifies systematic classification errors vs detection failures.

6. **Per-image precision/recall/F1** — Computed per image rather than per class, then grouped by weather, time of day, and object size to find failure clusters. This is where Phase 1 connects to Phase 3:
   - Rainy + night + tiny objects has the highest failure rate (37.1%)
   - Clear + daytime (the dominant condition in training) has much lower failure rates
   - This confirms the Phase 1 finding that the dataset is biased toward clear/daytime conditions

7. **Correlation analysis** — Pearson and Spearman correlations between image-level features and recall:
   - Object area vs recall: r=0.27 (larger objects are easier to detect)
   - GT count vs recall: r=-0.26 (crowded scenes hurt recall)
   - Brightness vs recall: r=0.16 (darker images are harder)
   - Occlusion rate vs recall: r=-0.08 (occluded objects slightly harder)

8. **Per-class failure clustering** — Groups failures by (class, weather, time-of-day) to find class-specific weak spots. For example, riders at night have higher failure rates than cars at night, because riders are smaller and rarer.

**Phase 1 to Phase 3 connection:**

The dashboard explicitly links data analysis findings to model performance:
- Class imbalance (Phase 1) directly predicts AP@50 (Phase 3) — rare classes perform worse
- High occlusion rates (Phase 1) correlate with lower AP for those classes (Phase 3)
- Weather/time bias in training data (Phase 1) maps to higher failure rates in underrepresented conditions (Phase 3)

**Data-driven improvement suggestions** are generated from the evaluation results — not hardcoded — based on patterns like: low AP + few training samples suggests oversampling, AP_small << AP_large suggests multi-scale augmentation, high failure rate for a weather condition suggests targeted augmentation.

```bash
# Run inference on val set
uv run python -m src.evaluation.run_inference

# Compute and cache eval metrics
uv run python -m src.evaluation.metrics
```

## Project Structure

```
src/
├── parser.py                  # JSON label parsing + drivable area extraction
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

## Given More Time

- **Class-balanced training** — focal loss or oversampling for rare classes (`train`: 170 samples, `motor`: 452) to close the AP gap with dominant classes
- **Targeted augmentation** — synthetic rain/night transforms and copy-paste augmentation for small objects, addressing the 37% failure rate in rainy+night conditions
- **Higher resolution training** — train at native 1280x720 to improve small object AP (currently 13.4% vs 61.6% for large)
- **Hard example mining** — use failure clusters to oversample the hardest images during training
- **TIDE error decomposition** — break AP loss into classification, localization, duplicate, and missed detection errors
- **Confidence calibration** — verify prediction scores are well-calibrated, critical for safety applications
- **Interactive failure browser** — click failure cluster cells to see the actual failing images
- **Model comparison view** — side-by-side evaluation across training runs
