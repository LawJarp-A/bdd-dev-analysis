"""Train RF-DETR on BDD100K with custom dataloader.

Uses RF-DETR's training recipe (layer-wise LR decay, cosine schedule with
warmup, EMA, AMP, gradient accumulation, gradient clipping).
"""

import argparse
import math
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from rfdetr.main import populate_args
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.misc import NestedTensor
from rfdetr.util.utils import ModelEma

from src.parser import DETECTION_CLASSES
from src.training.dataset import BDD100KDataset, collate_fn

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RF-DETR on BDD100K")
    p.add_argument("--model", default="large", choices=["base", "small", "medium", "large"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of training data (e.g. 0.01 for 1%%)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--name", type=str, default="rfdetr_bdd100k",
                   help="Run name for output directory")
    p.add_argument("--checkpoint-interval", type=int, default=5)
    return p.parse_args()


def _get_model(size: str):
    if size == "base":
        from rfdetr import RFDETRBase
        return RFDETRBase()
    if size == "small":
        from rfdetr import RFDETRSmall
        return RFDETRSmall()
    if size == "medium":
        from rfdetr import RFDETRMedium
        return RFDETRMedium()
    from rfdetr import RFDETRLarge
    return RFDETRLarge()


def _build_lr_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: float):
    """Cosine LR schedule with linear warmup."""
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    output_dir = RUNS_DIR / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading RF-DETR ({args.model}) ...")
    rfdetr = _get_model(args.model)

    # Head has num_classes+1 (extra no-object slot); SetCriterion adds +1 internally
    n_classes = len(DETECTION_CLASSES)
    rfdetr.model.reinitialize_detection_head(n_classes + 1)

    model = rfdetr.model.model
    model.to(device)
    model.train()
    img_size = rfdetr.model.resolution

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,} ({n_params / 1e6:.1f}M)")

    rfdetr_args = populate_args(
        num_classes=n_classes,
        device=str(device),
        dec_layers=rfdetr.model.args.dec_layers,
        aux_loss=rfdetr.model.args.aux_loss,
        group_detr=rfdetr.model.args.group_detr,
        segmentation_head=False,
    )
    criterion, _ = build_criterion_and_postprocessors(rfdetr_args)
    criterion.to(device)

    rfdetr_args.lr = args.lr
    rfdetr_args.lr_component_decay = 1.0
    rfdetr_args.lr_vit_layer_decay = 0.8
    param_dicts = get_param_dict(rfdetr_args, model)
    param_dicts = [p for p in param_dicts if p["params"].requires_grad]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)

    dataset = BDD100KDataset(split="train", img_size=img_size, fraction=args.fraction)
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(dataset) // effective_batch)
    lr_scheduler = _build_lr_scheduler(optimizer, steps_per_epoch, args.epochs, args.warmup_epochs)

    ema = ModelEma(model, decay=0.9997)
    scaler = GradScaler(enabled=use_amp)

    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, effective_batch, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        collate_fn=collate_fn, num_workers=2)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema_model" in ckpt:
            ema.module.load_state_dict(ckpt["ema_model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}")

    print(f"\nTraining RF-DETR ({args.model}) on BDD100K")
    print(f"  Dataset:       {len(dataset)} images"
          + (f" ({args.fraction:.0%} subset)" if args.fraction < 1 else ""))
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {effective_batch} (bs={args.batch_size} x accum={args.grad_accum})")
    print(f"  Steps/epoch:   {steps_per_epoch}")
    print(f"  LR:            {args.lr}  (cosine, warmup={args.warmup_epochs} epochs)")
    print(f"  Resolution:    {img_size}")
    print(f"  Device:        {device}")
    print(f"  AMP:           {use_amp}")
    print(f"  Output:        {output_dir}\n")

    weight_dict = criterion.weight_dict
    sub_batch = args.batch_size  # per grad-accum step

    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.train()
        epoch_loss = 0.0

        for step, (samples, targets) in enumerate(loader):
            samples = samples.to(device)

            for i in range(args.grad_accum):
                s = i * sub_batch
                e = s + sub_batch
                sub_samples_tensors = samples.tensors[s:e]
                sub_mask = samples.mask[s:e]

                sub_samples = NestedTensor(sub_samples_tensors, sub_mask)
                sub_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[s:e]]

                with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                    outputs = model(sub_samples, sub_targets)
                    loss_dict = criterion(outputs, sub_targets)
                    loss = sum(
                        (1.0 / args.grad_accum) * loss_dict[k] * weight_dict[k]
                        for k in loss_dict if k in weight_dict
                    )

                scaler.scale(loss).backward()

            loss_value = loss.item() * args.grad_accum
            if not math.isfinite(loss_value):
                print(f"  WARNING: loss is {loss_value}, skipping step")
                optimizer.zero_grad()
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            ema.update(model)
            epoch_loss += loss_value
            if (step + 1) % 10 == 0 or step == 0:
                avg = epoch_loss / (step + 1)
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  [Epoch {epoch+1}/{args.epochs}] "
                      f"Step {step+1:>4d}/{steps_per_epoch}  "
                      f"loss={loss_value:.4f}  avg={avg:.4f}  lr={lr_now:.2e}")

        avg_epoch_loss = epoch_loss / max(steps_per_epoch, 1)
        print(f"  Epoch {epoch+1} complete — avg loss: {avg_epoch_loss:.4f}\n")

        ckpt_data = {
            "model": model.state_dict(),
            "ema_model": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "args": rfdetr.model.args,
        }

        torch.save(ckpt_data, output_dir / "checkpoint.pth")

        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(ckpt_data, output_dir / f"checkpoint{epoch:04d}.pth")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ckpt_data["best_loss"] = best_loss
            torch.save(ckpt_data, output_dir / "checkpoint_best.pth")
            print(f"  New best loss: {best_loss:.4f}")

    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
