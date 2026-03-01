# Phase 2: Model Training

## Model Choice: RF-DETR Large

[RF-DETR](https://github.com/roboflow/rf-detr) — real-time DETR variant with DINOv2 backbone, 33.9M params, 56.5 AP on COCO.

**Why RF-DETR for BDD100K:**

| BDD100K Challenge | RF-DETR Advantage |
|---|---|
| 87.4% small objects | Multi-scale training + deformable attention |
| 5,402:1 class imbalance | Bipartite matching loss, no NMS bias |
| High occlusion (rider 89%) | DINOv2 self-attention captures global context |
| Real-time requirement | 6.8ms inference on T4 |

## Training Configuration

Fine-tuning from COCO pretrained weights. 6 of 10 BDD100K classes overlap with COCO.

| Parameter | Value |
|---|---|
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| LR schedule | Cosine with 3-epoch linear warmup |
| Backbone LR decay | Layer-wise 0.8x per ViT layer |
| AMP | bfloat16 mixed precision |
| EMA | decay=0.9997 |
| Gradient clipping | max_norm=0.1 |
| Resolution | Model default (560px for Large) |

## Custom Training Pipeline

Instead of using RF-DETR's built-in training, we wrote a custom loop (`src/training/train.py`) that:

1. Loads data through our `BDD100KDataset` with albumentations augmentations
2. Outputs normalized cxcywh boxes via custom `collate_fn` producing `NestedTensor`
3. Uses RF-DETR's `SetCriterion` (Hungarian matching) for loss computation
4. Replicates RF-DETR's optimizer recipe: layer-wise LR decay via `get_param_dict`
5. Supports gradient accumulation, AMP, EMA, and checkpoint saving

## How to Run

```bash
# Full training
uv run python -m src.training.train --epochs 50 --batch-size 16

# Quick test
uv run python -m src.training.train --epochs 1 --fraction 0.01 --batch-size 2

# Resume from checkpoint
uv run python -m src.training.train --resume runs/rfdetr_bdd100k/checkpoint.pth
```

Checkpoints saved to `runs/<name>/` (latest, periodic, best).
