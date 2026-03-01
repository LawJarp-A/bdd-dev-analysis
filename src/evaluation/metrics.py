"""Evaluation metrics and failure analysis for BDD100K predictions."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import stats

from src.compute_image_metrics import METRICS_PATH
from src.parser import DATA_DIR, DETECTION_CLASSES, parse_labels

COCO_ANN = DATA_DIR / "bdd100k_coco" / "valid" / "_annotations.coco.json"
PRED_JSON = DATA_DIR / "predictions" / "val_predictions.json"
EVAL_PKL = DATA_DIR / "predictions" / "eval_results.pkl"

_OVERALL_KEYS = [
    "mAP5095", "mAP50", "mAP75",
    "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_500",
    "AR_small", "AR_medium", "AR_large",
]
_RECALL_THRESHOLDS = np.linspace(0, 1, 101)
_IMG_AREA = 1280 * 720

Box = list[float]


# --- Helpers ---


def _safe_mean(arr: np.ndarray) -> float:
    valid = arr[arr > -1]
    return float(np.mean(valid)) if valid.size > 0 else 0.0


def _iou(box1: Box, box2: Box) -> float:
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(
    pred_boxes: list[Box], gt_boxes: np.ndarray, iou_thresh: float = 0.5
) -> dict[int, int]:
    """Greedy-match predictions to GT. Returns {gt_idx: pred_idx}."""
    matched: dict[int, int] = {}
    for p_idx, pb in enumerate(pred_boxes):
        best_iou, best_gt = 0.0, -1
        for g_idx, gb in enumerate(gt_boxes):
            if g_idx in matched:
                continue
            iou_val = _iou(pb, list(gb))
            if iou_val > best_iou:
                best_iou, best_gt = iou_val, g_idx
        if best_iou >= iou_thresh and best_gt >= 0:
            matched[best_gt] = p_idx
    return matched


def _load_preds_by_image(conf_thresh: float) -> dict[str, list[dict]]:
    with open(PRED_JSON) as f:
        preds = json.load(f)
    with open(COCO_ANN) as f:
        coco = json.load(f)
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

    by_img: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        if p["score"] < conf_thresh:
            continue
        x, y, w, h = p["bbox"]
        by_img[id_to_name.get(p["image_id"], "")].append(
            {"box": [x, y, x + w, y + h], "class": DETECTION_CLASSES[p["category_id"]], "score": p["score"]}
        )
    for name in by_img:
        by_img[name].sort(key=lambda d: d["score"], reverse=True)
    return dict(by_img)


# --- COCO Metrics ---


def compute_coco_metrics() -> dict:
    coco_gt = COCO(str(COCO_ANN))
    coco_dt = coco_gt.loadRes(str(PRED_JSON))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    precision = ev.eval["precision"]  # (T=10, R=101, K, A=4, M=3)
    per_class, pr_curves = [], {}
    for k, cat_id in enumerate(ev.params.catIds):
        ap50_curve = precision[0, :, k, 0, 2]
        cls_name = DETECTION_CLASSES[cat_id]
        per_class.append({
            "class": cls_name,
            "AP50": round(_safe_mean(ap50_curve), 4),
            "AP5095": round(_safe_mean(precision[:, :, k, 0, 2]), 4),
            "AP_small": round(_safe_mean(precision[:, :, k, 1, 2]), 4),
            "AP_medium": round(_safe_mean(precision[:, :, k, 2, 2]), 4),
            "AP_large": round(_safe_mean(precision[:, :, k, 3, 2]), 4),
            "num_gt": len(coco_gt.getAnnIds(catIds=[cat_id])),
        })
        pr_curves[cls_name] = {
            "precision": ap50_curve.tolist(),
            "recall": _RECALL_THRESHOLDS.tolist(),
        }

    return {
        "overall": dict(zip(_OVERALL_KEYS, ev.stats)),
        "per_class": pd.DataFrame(per_class),
        "pr_curves": pr_curves,
    }


# --- Per-Image & Per-Class Stats (single pass) ---


def _compute_all_stats(
    gt_df: pd.DataFrame, iou_thresh: float = 0.5, conf_thresh: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single pass over val images producing confusion matrix, per-image stats, and per-class clusters."""
    pred_by_img = _load_preds_by_image(conf_thresh)
    val_df = gt_df[gt_df["split"] == "val"]

    classes = DETECTION_CLASSES + ["background"]
    cm = pd.DataFrame(0, index=DETECTION_CLASSES, columns=classes)
    img_rows, class_rows = [], []

    for img_name, img_gts in val_df.groupby("image_name"):
        gt_boxes = img_gts[["x1", "y1", "x2", "y2"]].values
        preds = pred_by_img.get(img_name, [])
        pred_boxes = [p["box"] for p in preds]
        first = img_gts.iloc[0]

        # Confusion matrix — needs pred-to-GT mapping
        match_map = _greedy_match(pred_boxes, gt_boxes, iou_thresh)
        for gt_idx, pred_idx in match_map.items():
            cm.loc[img_gts.iloc[gt_idx]["category"], preds[pred_idx]["class"]] += 1
        for g_idx in range(len(img_gts)):
            if g_idx not in match_map:
                cm.loc[img_gts.iloc[g_idx]["category"], "background"] += 1

        # Per-image stats
        tp = len(match_map)
        fp, fn = len(pred_boxes) - tp, len(gt_boxes) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        img_rows.append({
            "image_name": img_name,
            "gt_count": len(gt_boxes), "pred_count": len(pred_boxes),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0, 4),
            "weather": first["weather"], "scene": first["scene"], "timeofday": first["timeofday"],
            "avg_area": img_gts["area"].mean(), "occlusion_rate": img_gts["occluded"].mean(),
        })

        # Per-class stats for failure clustering
        for cat, cat_gts in img_gts.groupby("category"):
            cat_gt_boxes = cat_gts[["x1", "y1", "x2", "y2"]].values
            cat_pred_boxes = [p["box"] for p in preds if p["class"] == cat]
            cat_matched = _greedy_match(cat_pred_boxes, cat_gt_boxes, iou_thresh)
            cat_recall = len(cat_matched) / len(cat_gt_boxes) if len(cat_gt_boxes) > 0 else 0.0
            class_rows.append({
                "image_name": img_name, "category": cat,
                "weather": first["weather"], "timeofday": first["timeofday"],
                "gt_count": len(cat_gt_boxes), "tp": len(cat_matched),
                "recall": round(cat_recall, 4),
            })

    per_image = pd.DataFrame(img_rows)
    if METRICS_PATH.exists():
        per_image = per_image.merge(pd.read_csv(METRICS_PATH), on="image_name", how="left")

    # Cluster per-class failures by condition
    per_class_df = pd.DataFrame(class_rows)
    per_class_df["is_failure"] = per_class_df["recall"] < 0.5
    per_class_clusters = (
        per_class_df.groupby(["category", "weather", "timeofday"])
        .agg(failure_rate=("is_failure", "mean"), mean_recall=("recall", "mean"), n_images=("image_name", "count"))
        .round(4).sort_values("failure_rate", ascending=False).reset_index()
    )
    per_class_clusters = per_class_clusters[per_class_clusters["n_images"] >= 5]

    return cm, per_image, per_class_clusters


# --- Build / Load Cache ---


def build_cache() -> dict:
    print("Computing COCO metrics...")
    coco_metrics = compute_coco_metrics()

    print("Parsing GT labels...")
    gt_df = parse_labels(split="val")

    print("Computing confusion matrix, per-image stats, and per-class clusters...")
    cm, per_image, per_class_clusters = _compute_all_stats(gt_df)

    # Build failure analysis
    print("Building failure analysis...")
    img_stats = (
        gt_df.groupby("image_name")
        .agg(dominant_class=("category", lambda x: x.value_counts().index[0]),
             avg_area=("area", "mean"), occlusion_rate=("occluded", "mean"))
        .reset_index()
    )
    failure_df = per_image.merge(img_stats, on="image_name", how="left", suffixes=("", "_gt"))
    failure_df["is_failure"] = failure_df["recall"] < 0.5
    failure_df["size_bucket"] = pd.cut(
        failure_df["avg_area"] / _IMG_AREA,
        bins=[0, 0.01, 0.05, 0.2, 1.0], labels=["tiny", "small", "medium", "large"],
    )

    # Cluster overall failures
    clusters = (
        failure_df.groupby(["weather", "timeofday", "size_bucket"])
        .agg(failure_rate=("is_failure", "mean"), mean_recall=("recall", "mean"),
             mean_precision=("precision", "mean"), n_images=("image_name", "count"))
        .round(4).sort_values("failure_rate", ascending=False).reset_index()
    )
    clusters = clusters[clusters["n_images"] >= 5]

    # Correlation table
    features = ["avg_area", "occlusion_rate", "gt_count"]
    if "blur_score" in failure_df.columns:
        features += ["blur_score", "mean_brightness", "contrast"]
    corr_rows = []
    for feat in features:
        valid = failure_df[[feat, "recall"]].dropna()
        if len(valid) < 10:
            continue
        pr, pp = stats.pearsonr(valid[feat], valid["recall"])
        sr, sp = stats.spearmanr(valid[feat], valid["recall"])
        corr_rows.append({
            "feature": feat, "pearson_r": round(pr, 4), "pearson_p": round(pp, 4),
            "spearman_r": round(sr, 4), "spearman_p": round(sp, 4),
        })

    results = {
        "coco_metrics": coco_metrics, "confusion_matrix": cm,
        "per_image": per_image, "failure_df": failure_df,
        "clusters": clusters, "correlation_table": pd.DataFrame(corr_rows),
        "per_class_clusters": per_class_clusters,
    }
    EVAL_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {EVAL_PKL}")
    return results


def load_cache() -> dict | None:
    if EVAL_PKL.exists():
        with open(EVAL_PKL, "rb") as f:
            return pickle.load(f)
    return None


if __name__ == "__main__":
    build_cache()
