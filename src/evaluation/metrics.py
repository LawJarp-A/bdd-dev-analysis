"""Evaluation metrics and failure analysis for BDD100K predictions."""

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
    "mAP5095", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_500", "AR_small", "AR_medium", "AR_large",
]
_RECALL_THRESHOLDS = np.linspace(0, 1, 101)


# --- Helpers ---

def _safe_mean(arr: np.ndarray) -> float:
    valid = arr[arr > -1]
    return float(np.mean(valid)) if valid.size > 0 else 0.0


def _iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union if union > 0 else 0.0


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
        by_img[id_to_name.get(p["image_id"], "")].append({
            "box": [x, y, x + w, y + h], "class": DETECTION_CLASSES[p["category_id"]], "score": p["score"],
        })
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
        pr_curves[cls_name] = {"precision": ap50_curve.tolist(), "recall": _RECALL_THRESHOLDS.tolist()}

    return {"overall": dict(zip(_OVERALL_KEYS, ev.stats)), "per_class": pd.DataFrame(per_class), "pr_curves": pr_curves}


# --- Confusion Matrix ---

def compute_confusion_matrix(gt_df: pd.DataFrame, iou_thresh=0.5, conf_thresh=0.5) -> pd.DataFrame:
    pred_by_img = _load_preds_by_image(conf_thresh)
    classes = DETECTION_CLASSES + ["background"]
    cm = pd.DataFrame(0, index=DETECTION_CLASSES, columns=classes)

    for img_name, img_gts in gt_df[gt_df["split"] == "val"].groupby("image_name"):
        gt_boxes = img_gts[["x1", "y1", "x2", "y2"]].values
        gt_matched = set()
        for pred in pred_by_img.get(img_name, []):
            best_iou, best_idx = 0, -1
            for idx, gb in enumerate(gt_boxes):
                if idx not in gt_matched:
                    iou_val = _iou(pred["box"], gb)
                    if iou_val > best_iou:
                        best_iou, best_idx = iou_val, idx
            if best_iou >= iou_thresh and best_idx >= 0:
                gt_matched.add(best_idx)
                cm.loc[img_gts.iloc[best_idx]["category"], pred["class"]] += 1
        for idx in range(len(img_gts)):
            if idx not in gt_matched:
                cm.loc[img_gts.iloc[idx]["category"], "background"] += 1
    return cm


# --- Per-Image Stats ---

def compute_per_image_stats(gt_df: pd.DataFrame, iou_thresh=0.5, conf_thresh=0.5) -> pd.DataFrame:
    pred_by_img = _load_preds_by_image(conf_thresh)
    rows = []
    for img_name, img_gts in gt_df[gt_df["split"] == "val"].groupby("image_name"):
        gt_boxes = img_gts[["x1", "y1", "x2", "y2"]].values
        pred_boxes = [p["box"] for p in pred_by_img.get(img_name, [])]

        # Greedy match
        gt_matched = set()
        for pb in pred_boxes:
            best_iou, best_idx = 0, -1
            for i, gb in enumerate(gt_boxes):
                if i not in gt_matched:
                    iou_val = _iou(pb, gb)
                    if iou_val > best_iou:
                        best_iou, best_idx = iou_val, i
            if best_iou >= iou_thresh and best_idx >= 0:
                gt_matched.add(best_idx)

        tp = len(gt_matched)
        fn, fp = len(gt_boxes) - tp, len(pred_boxes) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        first = img_gts.iloc[0]
        rows.append({
            "image_name": img_name, "gt_count": len(gt_boxes), "pred_count": len(pred_boxes),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "weather": first["weather"], "scene": first["scene"], "timeofday": first["timeofday"],
            "avg_area": img_gts["area"].mean(), "occlusion_rate": img_gts["occluded"].mean(),
        })

    result = pd.DataFrame(rows)
    if METRICS_PATH.exists():
        result = result.merge(pd.read_csv(METRICS_PATH), on="image_name", how="left")
    return result


# --- Failure Analysis ---

def build_failure_df(per_image: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    img_stats = (
        gt_df[gt_df["split"] == "val"].groupby("image_name")
        .agg(dominant_class=("category", lambda x: x.value_counts().index[0]),
             avg_area=("area", "mean"), occlusion_rate=("occluded", "mean"))
        .reset_index()
    )
    df = per_image.merge(img_stats, on="image_name", how="left", suffixes=("", "_gt"))
    df["is_failure"] = df["recall"] < 0.5
    df["size_bucket"] = pd.cut(
        df["avg_area"] / (1280 * 720), bins=[0, 0.01, 0.05, 0.2, 1.0],
        labels=["tiny", "small", "medium", "large"],
    )
    return df


def cluster_failures(failure_df: pd.DataFrame) -> pd.DataFrame:
    groups = failure_df.groupby(["weather", "timeofday", "size_bucket"]).agg(
        failure_rate=("is_failure", "mean"), mean_recall=("recall", "mean"),
        mean_precision=("precision", "mean"), n_images=("image_name", "count"),
    ).round(4).sort_values("failure_rate", ascending=False).reset_index()
    return groups[groups["n_images"] >= 5]


def phase1_correlation_table(failure_df: pd.DataFrame) -> pd.DataFrame:
    features = ["avg_area", "occlusion_rate", "gt_count"]
    if "blur_score" in failure_df.columns:
        features += ["blur_score", "mean_brightness", "contrast"]
    rows = []
    for feat in features:
        valid = failure_df[[feat, "recall"]].dropna()
        if len(valid) < 10:
            continue
        pr, pp = stats.pearsonr(valid[feat], valid["recall"])
        sr, sp = stats.spearmanr(valid[feat], valid["recall"])
        rows.append({"feature": feat, "pearson_r": round(pr, 4), "pearson_p": round(pp, 4),
                      "spearman_r": round(sr, 4), "spearman_p": round(sp, 4)})
    return pd.DataFrame(rows)


# --- Build / Load Cache ---

def build_cache() -> dict:
    print("Computing COCO metrics...")
    coco_metrics = compute_coco_metrics()

    print("Parsing GT labels...")
    gt_df = parse_labels(split="val")

    print("Computing confusion matrix...")
    cm = compute_confusion_matrix(gt_df)

    print("Computing per-image stats...")
    per_image = compute_per_image_stats(gt_df)

    print("Building failure analysis...")
    failure_df = build_failure_df(per_image, gt_df)
    clusters = cluster_failures(failure_df)
    corr_table = phase1_correlation_table(failure_df)

    results = {
        "coco_metrics": coco_metrics, "confusion_matrix": cm, "per_image": per_image,
        "failure_df": failure_df, "clusters": clusters, "correlation_table": corr_table,
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
