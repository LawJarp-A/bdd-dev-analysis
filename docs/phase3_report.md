# Phase 3: Evaluation

**Model:** RF-DETR Large, 20 epochs on full BDD100K, best checkpoint at epoch 13.

## Overall Metrics

| Metric | Value |
|---|---|
| mAP@50 | 0.596 |
| mAP@50:95 | 0.338 |
| AP Small | 0.134 |
| AP Medium | 0.386 |
| AP Large | 0.616 |

## Per-Class AP@50

| Class | AP@50 | Val Count |
|---|---|---|
| car | 0.814 | 102,506 |
| traffic sign | 0.739 | 34,908 |
| traffic light | 0.687 | 26,885 |
| bus | 0.685 | 1,597 |
| truck | 0.685 | 4,245 |
| person | 0.664 | 13,262 |
| motor | 0.566 | 452 |
| bike | 0.542 | 1,007 |
| rider | 0.513 | 649 |
| train | 0.063 | 15 |

## Phase 1 Predictions Validated

| Phase 1 Finding | Phase 3 Result |
|---|---|
| 87.4% tiny boxes | AP_small=0.134 vs AP_large=0.616 (4.6x gap) |
| rider 89% occluded | rider AP@50=0.513 (lowest common class) |
| train 0.01% of data | train AP@50=0.063 |
| bike 84% occluded | bike AP@50=0.542 |
| Night images harder | Night F1=0.693 vs Daytime F1=0.756 |

## Safety-Critical Scenarios

| Scenario | Images | Failure Rate | Mean Recall |
|---|---|---|---|
| Tiny VRU night/rain | 320 | 41.6% | 0.519 |
| Occluded pedestrian near cars | 363 | 33.1% | 0.547 |
| Dark with VRU | 96 | 35.4% | 0.551 |
| Truncated person at edge | 328 | 14.9% | 0.629 |

## Improvement Suggestions

1. Oversample rare classes (train/bike/motor) + copy-paste augmentation
2. Increase training resolution for small object detection
3. Night/weather-specific augmentations (gamma jitter, rain overlay)
4. Random erasing augmentation for occlusion robustness
5. Mine hard examples from failure analysis for targeted training

## Running Evaluation

```bash
uv run python -m src.evaluation.run_inference   # inference on val set
uv run python -m src.evaluation.metrics          # compute metrics
uv run streamlit run src/dashboard.py            # view in dashboard
```
