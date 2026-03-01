"""Streamlit dashboard for BDD100K data analysis."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package is importable
# when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src import analysis  # noqa: E402
from src.compute_image_metrics import METRICS_PATH  # noqa: E402
from src.evaluation.metrics import COCO_ANN, PRED_JSON, load_cache as load_eval_cache  # noqa: E402
from src.parser import (  # noqa: E402
    DETECTION_CLASSES,
    IMAGE_DIRS,
    annotate_ego_lane,
    parse_drivable_areas,
    parse_labels,
)


def _show_fig(fig: plt.Figure) -> None:
    """Display a matplotlib figure in Streamlit and close it to free memory."""
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the parsed dataset with ego lane annotations."""
    df = parse_labels(split="all")
    return annotate_ego_lane(df, parse_drivable_areas(split="all"))


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    """Load pre-computed image quality metrics if available."""
    return pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None


# ---------------------------------------------------------------------------
# Overview Tab
# ---------------------------------------------------------------------------


def overview_tab(df: pd.DataFrame) -> None:
    """Overview page with summary stats and key distributions."""
    st.header("Dataset Overview")

    n_annotations = len(df)
    n_images = df["image_name"].nunique()
    avg_per_image = n_annotations / max(n_images, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Annotations", f"{n_annotations:,}")
    col2.metric("Unique Images", f"{n_images:,}")
    col3.metric("Classes", df["category"].nunique())
    col4.metric("Avg Objects/Image", f"{avg_per_image:.1f}")

    st.subheader("Class Distribution")
    _show_fig(analysis.plot_class_distribution(df))

    st.subheader("Co-occurrence Matrix")
    _show_fig(analysis.plot_cooccurrence(df))

    st.subheader("Train/Val Split Balance")
    chi2_result = analysis.split_balance_chi2(df)
    st.write(
        f"Chi-squared: **{chi2_result['chi2']:.2f}**, p-value: **{chi2_result['p_value']:.2e}**, dof: {chi2_result['dof']}"
    )
    if chi2_result["p_value"] < 0.05:
        st.warning("Class distribution differs significantly between train and val.")
    else:
        st.success("Train and val splits are well-balanced.")

    st.subheader("Attribute Summary")
    for attr in ["weather", "scene", "timeofday"]:
        counts = df.drop_duplicates(subset="image_name")[attr].value_counts()
        st.write(f"**{attr.title()}**")
        st.dataframe(counts.reset_index().rename(columns={"index": attr, attr: "count", "count": "images"}), hide_index=True)


# ---------------------------------------------------------------------------
# Class Deep Dive Tab
# ---------------------------------------------------------------------------


def class_deep_dive_tab(df: pd.DataFrame) -> None:
    """Per-class analysis page."""
    st.header("Class Deep Dive")

    st.subheader("Objects per Image Distribution")
    _show_fig(analysis.plot_objects_per_image(df))

    st.subheader("Occlusion & Truncation by Class")
    _show_fig(analysis.plot_occlusion_truncation(df))

    st.divider()

    selected_class = st.selectbox("Select Class", sorted(df["category"].unique()))
    class_df = df[df["category"] == selected_class]

    col1, col2, col3 = st.columns(3)
    col1.metric("Count", f"{len(class_df):,}")
    col2.metric("Avg Area", f"{class_df['area'].mean():,.0f} px\u00b2")
    col3.metric("Avg Aspect Ratio", f"{class_df['aspect_ratio'].mean():.2f}")

    st.subheader("Bounding Box Size Distribution")
    _show_fig(analysis.plot_bbox_area_distribution(class_df))

    st.subheader("Spatial Heatmap")
    _show_fig(analysis.plot_spatial_heatmap(df, category=selected_class))


# ---------------------------------------------------------------------------
# Image Viewer (shared by anomalies, safety-critical, and sample browser)
# ---------------------------------------------------------------------------


def _render_image_with_boxes(
    img_path: Path,
    img_data: pd.DataFrame,
    highlight_indices: set | None = None,
) -> plt.Figure:
    """Draw bounding boxes on an image. Highlighted indices are shown in red; others faded."""
    img_np = np.array(Image.open(img_path))

    fig, ax = plt.subplots(1, figsize=(14, 8))
    ax.imshow(img_np)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    class_colors = {cls: colors[i] for i, cls in enumerate(DETECTION_CLASSES)}

    for idx, row in img_data.iterrows():
        if highlight_indices is None:
            # Normal mode: all boxes shown with class-specific colors
            is_highlighted = True
            color = class_colors.get(row["category"], (1, 0, 0))
            linewidth = 2
            alpha = 0.7
        elif idx in highlight_indices:
            # Highlight mode: flagged box in bold red
            is_highlighted = True
            color = (1, 0, 0)
            linewidth = 3
            alpha = 0.9
        else:
            # Highlight mode: non-flagged box in faded gray
            is_highlighted = False
            color = (0.6, 0.6, 0.6)
            linewidth = 1
            alpha = 0.3

        rect = patches.Rectangle(
            (row["x1"], row["y1"]),
            row["width"],
            row["height"],
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            alpha=alpha,
        )
        ax.add_patch(rect)

        if is_highlighted:
            ax.text(
                row["x1"],
                row["y1"] - 5,
                row["category"],
                color="white",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
            )

    ax.set_title(f"{img_path.name} \u2014 {len(img_data)} objects")
    ax.axis("off")
    plt.tight_layout()
    return fig


def _show_anomaly_sample(
    anomaly_df: pd.DataFrame,
    full_df: pd.DataFrame,
    key: str,
) -> None:
    """Show a random sample image with anomalous bboxes highlighted in red."""
    if anomaly_df.empty:
        return
    if st.button("Show another sample", key=key):
        sample_row = anomaly_df.sample(1).iloc[0]
        img_name = sample_row["image_name"]
        split = sample_row["split"]
        img_path = IMAGE_DIRS[split] / img_name
        img_data = full_df[full_df["image_name"] == img_name]

        # Highlight only the anomalous boxes in this image
        anomalous_in_image = anomaly_df[anomaly_df["image_name"] == img_name]
        highlight_set = set(anomalous_in_image.index)

        if img_path.exists():
            _show_fig(
                _render_image_with_boxes(img_path, img_data, highlight_indices=highlight_set)
            )
            st.caption(
                f"Image: {img_name} | "
                f"{len(highlight_set)} flagged box(es) shown in red, others in gray"
            )
        else:
            st.error(f"Image not found: {img_path}")


# ---------------------------------------------------------------------------
# Anomalies Tab
# ---------------------------------------------------------------------------


def _anomaly_category_summary(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    """Group anomalous boxes by category with counts, sorted descending."""
    return (
        anomaly_df.groupby("category")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )


def anomalies_tab(df: pd.DataFrame) -> None:
    """Anomaly detection page with explanations and highlighted sample images."""
    st.header("Anomalies")

    total = max(len(df), 1)

    # --- Per-class outliers ---
    st.subheader("Per-Class Size Outliers (5th / 95th percentile)")
    st.write(
        "Boxes that are unusually small or large **for their class** "
        "(e.g. a 50 px\u00b2 car is an outlier, but a 50 px\u00b2 traffic light isn't). "
        "Could be annotation errors, extreme distances, or unusual viewpoints."
    )
    outliers = analysis.per_class_outliers(df, column="area")
    if not outliers.empty:
        summary = (
            outliers.groupby(["category", "outlier_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        st.dataframe(summary)
        _show_anomaly_sample(outliers, df, key="outlier_sample")

    # --- Tiny & huge boxes ---
    st.subheader("Extreme Size Boxes")
    tiny = analysis.tiny_boxes(df)
    huge = analysis.huge_boxes(df)
    st.write(
        f"**{len(tiny):,}** boxes ({len(tiny) / total * 100:.1f}%) are under 1% of image area "
        f"(far-range objects). **{len(huge):,}** exceed 80% (very close-range). "
        f"Models need multi-scale features (e.g. FPN) to handle both extremes."
    )
    if not tiny.empty:
        st.dataframe(_anomaly_category_summary(tiny))
        _show_anomaly_sample(tiny, df, key="tiny_sample")

    # --- Extreme aspect ratios ---
    st.subheader("Extreme Aspect Ratios (< 0.1 or > 10)")
    extreme = analysis.extreme_aspect_ratios(df)
    st.write(
        f"**{len(extreme):,}** boxes ({len(extreme) / total * 100:.2f}%) with extreme aspect ratios. "
        f"Usually partial objects at frame edges or annotation artifacts. "
        f"Negligible training impact, but worth reviewing."
    )
    if not extreme.empty:
        st.dataframe(_anomaly_category_summary(extreme))
        _show_anomaly_sample(extreme, df, key="extreme_sample")


# ---------------------------------------------------------------------------
# Safety-Critical Edge Cases Tab
# ---------------------------------------------------------------------------


def _safety_case_section(
    title: str,
    explanation: str,
    case_df: pd.DataFrame,
    full_df: pd.DataFrame,
    key: str,
) -> None:
    """Render one safety-critical edge case section."""
    st.subheader(title)
    st.write(explanation)

    if case_df.empty:
        st.info("No matching cases found in the dataset.")
        return

    n_images = case_df["image_name"].nunique()
    n_boxes = len(case_df)
    st.write(f"**{n_images:,}** images, **{n_boxes:,}** flagged annotations")
    _show_anomaly_sample(case_df, full_df, key=key)


def safety_critical_tab(df: pd.DataFrame, metrics_df: pd.DataFrame | None) -> None:
    """Safety-critical edge cases relevant to autonomous driving."""
    st.header("Safety-Critical Edge Cases")
    st.write(
        "Scenarios where detection is hardest and failure costs are highest. "
        "Each case stacks multiple risk factors (small size, poor visibility, occlusion, etc.)."
    )

    ego_lane_only = st.checkbox(
        "Show only objects in the driving lane",
        value=False,
        help=(
            "Only flag objects whose foot point falls inside BDD100K's drivable area annotation "
            "(the vehicle's driving lane). Applies to: tiny road users, occluded pedestrians, blurry/dark frames."
        ),
    )
    ego_suffix = " **[Driving Lane Filter ON]**" if ego_lane_only else ""

    _safety_case_section(
        title="Tiny Road Users at Night / Rain",
        explanation=(
            "Pedestrians, riders, and cyclists under **0.05% of image area** (~21x21 px) "
            "in night or rain. Barely a handful of pixels in poor visibility — "
            "detectors have near-zero recall here." + ego_suffix
        ),
        case_df=analysis.tiny_vru_night_rain(df, ego_lane_only=ego_lane_only),
        full_df=df,
        key="safety_tiny_vru",
    )

    _safety_case_section(
        title="Occluded Tiny Pedestrians Near Cars",
        explanation=(
            "The worst-case ADAS scenario: pedestrians that are **tiny** (<0.1% area), "
            "**occluded**, near **cars**, and in **night or rain** — all at once." + ego_suffix
        ),
        case_df=analysis.occluded_pedestrian_near_cars(df, ego_lane_only=ego_lane_only),
        full_df=df,
        key="safety_occluded_ped",
    )

    _safety_case_section(
        title="Crowded Night Intersections",
        explanation=(
            "Night city scenes with **50+ objects**. High density pushes NMS to its limits — "
            "overlapping detections get suppressed and small objects between larger ones are missed."
        ),
        case_df=analysis.crowded_night_intersection(df),
        full_df=df,
        key="safety_crowded_night",
    )

    _safety_case_section(
        title="Truncated Pedestrians at Frame Edge",
        explanation=(
            "Truncated pedestrians **within 20 px of the frame boundary** — people entering or "
            "leaving the field of view. Detectors trained on full bodies struggle with partial views."
        ),
        case_df=analysis.truncated_person_edge(df),
        full_df=df,
        key="safety_truncated",
    )

    # Image-quality-based cases (require pre-computed metrics)
    if metrics_df is not None:
        _safety_case_section(
            title="Severely Blurry Frames with Pedestrians",
            explanation=(
                "Bottom **2% by sharpness** (Laplacian variance < 15) with pedestrians present. "
                "Edges and textures are destroyed — the features detectors rely on simply aren't there."
                + ego_suffix
            ),
            case_df=analysis.blurry_with_pedestrians(
                df, metrics_df, ego_lane_only=ego_lane_only
            ),
            full_df=df,
            key="safety_blurry",
        )

        _safety_case_section(
            title="Very Dark Frames with Road Users",
            explanation=(
                "Bottom **10% by brightness** (mean pixel < 20) with pedestrians, riders, or cyclists. "
                "Near-black frames where road users have almost no contrast — even human annotators struggle."
                + ego_suffix
            ),
            case_df=analysis.dark_with_vru(
                df, metrics_df, ego_lane_only=ego_lane_only
            ),
            full_df=df,
            key="safety_dark",
        )
    else:
        st.warning(
            "Image quality metrics not available. Run "
            "`python -m src.compute_image_metrics` to enable blur and darkness analysis."
        )

    # Rare condition combos
    st.subheader("Rarest Weather + Time-of-Day Combinations")
    st.write(
        "Under-represented conditions the model has seen least. "
        "Expect weaker performance on these slices — targeted evaluation recommended."
    )
    combos = analysis.rare_condition_combos(df)
    st.dataframe(combos)

    # Blind spots callout
    st.subheader("Detection Blind Spots")
    st.info(
        "**Unlabeled safety-relevant objects** (not in the 10 BDD100K detection classes):\n\n"
        "- **Animals** (deer, dogs) — common rural/suburban collision cause\n"
        "- **Road debris** (fallen cargo, tire fragments)\n"
        "- **Construction zones** (cones, barriers, workers)\n"
        "- **Emergency vehicles** with non-standard profiles\n"
        "- **Non-standard road users** (wheelchairs, strollers, shopping carts)\n"
        "- **Fallen pedestrians** on the road\n\n"
        "Production systems need additional models or anomaly detection to cover these."
    )


# ---------------------------------------------------------------------------
# Sample Browser Tab
# ---------------------------------------------------------------------------


def _select_images(df: pd.DataFrame, mode: str, category: str | None = None, n: int = 20) -> list[str]:
    """Return up to *n* image names based on the selected browse mode."""
    if mode == "Most Crowded":
        counts = df.groupby("image_name").size().sort_values(ascending=False)
        return counts.head(n).index.tolist()
    if mode == "Rare Classes":
        class_counts = df["category"].value_counts()
        rare_classes = class_counts.nsmallest(3).index.tolist()
        rare_df = df[df["category"].isin(rare_classes)]
        return rare_df["image_name"].unique()[:n].tolist()
    if mode == "Single-Class Images":
        img_classes = df.groupby("image_name")["category"].nunique()
        single = img_classes[img_classes == 1].index
        subset = df[df["image_name"].isin(single)]
        if category:
            subset = subset[subset["category"] == category]
        return subset["image_name"].unique()[:n].tolist()
    if mode == "Per-Class Outliers":
        outliers = analysis.per_class_outliers(df, column="area")
        if category:
            outliers = outliers[outliers["category"] == category]
        return outliers["image_name"].unique()[:n].tolist()
    if mode == "Highly Occluded":
        occluded = df[df["occluded"]]
        if category:
            occluded = occluded[occluded["category"] == category]
        counts = occluded.groupby("image_name").size().sort_values(ascending=False)
        return counts.head(n).index.tolist()
    # Default: Random sampling
    subset = df[df["category"] == category] if category else df
    sample_size = min(n, subset["image_name"].nunique())
    return subset["image_name"].drop_duplicates().sample(sample_size).tolist()


def sample_browser_tab(df: pd.DataFrame) -> None:
    """Browse sample images with bounding box overlays."""
    st.header("Sample Browser")

    col_mode, col_class = st.columns(2)
    with col_mode:
        mode = st.selectbox(
            "Browse Mode",
            ["Most Crowded", "Rare Classes", "Single-Class Images",
             "Per-Class Outliers", "Highly Occluded", "Random"],
        )
    with col_class:
        class_filter = st.selectbox(
            "Filter by Class (optional)",
            ["All"] + sorted(df["category"].unique()),
        )

    category = class_filter if class_filter != "All" else None
    image_names = _select_images(df, mode, category=category)

    if not image_names:
        st.warning("No images match the current selection.")
        return

    selected_img = st.selectbox("Select Image", image_names)
    img_data = df[df["image_name"] == selected_img]
    split = img_data["split"].iloc[0]
    img_path = IMAGE_DIRS[split] / selected_img

    if img_path.exists():
        _show_fig(_render_image_with_boxes(img_path, img_data))
    else:
        st.error(f"Image not found: {img_path}")

    st.subheader("Annotations")
    st.dataframe(
        img_data[["category", "x1", "y1", "x2", "y2", "area", "occluded", "truncated"]]
    )


# ---------------------------------------------------------------------------
# Evaluation Cache Loaders
# ---------------------------------------------------------------------------


@st.cache_data
def load_eval_results() -> dict | None:
    """Load pre-computed evaluation + failure analysis results."""
    return load_eval_cache()


@st.cache_data
def load_predictions() -> dict:
    """Load raw predictions grouped by image name for qualitative viz."""
    pred_csv = PRED_JSON.with_suffix(".csv")
    if not pred_csv.exists() and not PRED_JSON.exists():
        return {}

    with open(COCO_ANN) as f:
        coco = json.load(f)
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

    by_img = defaultdict(list)
    if pred_csv.exists():
        import csv
        with open(pred_csv) as f:
            for row in csv.DictReader(f):
                x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
                by_img[id_to_name.get(int(row["image_id"]), "")].append({
                    "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                    "class": DETECTION_CLASSES[int(row["category_id"])],
                    "score": float(row["score"]),
                })
    else:
        with open(PRED_JSON) as f:
            preds = json.load(f)
        for p in preds:
            x, y, w, h = p["bbox"]
            by_img[id_to_name.get(p["image_id"], "")].append({
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                "class": DETECTION_CLASSES[p["category_id"]],
                "score": p["score"],
            })
    return dict(by_img)


# ---------------------------------------------------------------------------
# Tab 6: Model Evaluation (Quantitative + Failure Analysis merged)
# ---------------------------------------------------------------------------


def _render_gt_vs_pred(img_path: Path, gt_df: pd.DataFrame, pred_list: list[dict]) -> plt.Figure:
    """Render image with GT (green) and predictions (orange=TP, red=FP), missed GT in yellow."""
    from src.evaluation.metrics import _iou

    img_np = np.array(Image.open(img_path))
    fig, ax = plt.subplots(1, figsize=(14, 8))
    ax.imshow(img_np)

    gt_boxes = gt_df[["x1", "y1", "x2", "y2"]].values
    preds_sorted = sorted(pred_list, key=lambda p: p["score"], reverse=True)

    # Greedy match predictions to GT
    gt_matched = set()
    pred_status = []
    for p in preds_sorted:
        pb = [p["x1"], p["y1"], p["x2"], p["y2"]]
        best_iou, best_idx = 0, -1
        for i, gb in enumerate(gt_boxes):
            if i in gt_matched:
                continue
            iou_val = _iou(pb, gb)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i
        is_tp = best_iou >= 0.5 and best_idx >= 0
        if is_tp:
            gt_matched.add(best_idx)
        pred_status.append((p, is_tp))

    # Draw GT boxes: green=matched, yellow=missed
    for i, (_, row) in enumerate(gt_df.iterrows()):
        matched = i in gt_matched
        color = (0, 0.8, 0) if matched else (1, 1, 0)
        label = f"{row['category']} [{'GT' if matched else 'MISSED'}]"
        rect = patches.Rectangle((row["x1"], row["y1"]), row["x2"] - row["x1"], row["y2"] - row["y1"],
                                  linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(row["x1"], row["y1"] - 5, label, color="white", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))

    # Draw predictions: orange=TP, red=FP
    n_shown_preds = 0
    for p, is_tp in pred_status:
        if p["score"] < 0.3:
            continue
        n_shown_preds += 1
        color = (1, 0.6, 0) if is_tp else (1, 0, 0)
        label = f"{p['class']} {p['score']:.2f}" + ("" if is_tp else " [FP]")
        rect = patches.Rectangle((p["x1"], p["y1"]), p["x2"] - p["x1"], p["y2"] - p["y1"],
                                  linewidth=2, edgecolor=color, facecolor="none",
                                  linestyle="--" if is_tp else "-")
        ax.add_patch(rect)
        ax.text(p["x2"], p["y1"] - 5, label, color="white", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))

    n_missed = len(gt_boxes) - len(gt_matched)
    ax.set_title(f"{img_path.name} | GT: {len(gt_boxes)}, Preds: {n_shown_preds}, Missed: {n_missed}")
    ax.axis("off")
    plt.tight_layout()
    return fig


def model_evaluation_tab(df: pd.DataFrame) -> None:
    """Combined model evaluation: quantitative metrics + failure analysis."""
    st.header("Model Evaluation")

    eval_results = load_eval_results()
    if eval_results is None:
        st.warning("Evaluation not yet computed. Run: `uv run python -m src.evaluation.run_inference` then `uv run python -m src.evaluation.metrics`")
        return

    coco = eval_results["coco_metrics"]
    overall = coco["overall"]

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("mAP@50", f"{overall['mAP50']:.3f}")
    c2.metric("mAP@50:95", f"{overall['mAP5095']:.3f}")
    c3.metric("AP Small", f"{overall['AP_small']:.3f}")
    c4.metric("AP Medium", f"{overall['AP_medium']:.3f}")
    c5.metric("AP Large", f"{overall['AP_large']:.3f}")

    # Per-class table
    st.subheader("Per-Class Performance")
    st.dataframe(coco["per_class"].sort_values("AP50", ascending=False), hide_index=True)

    # PR Curves
    st.subheader("Precision-Recall Curves")
    pr_class = st.selectbox("Class", ["All"] + DETECTION_CLASSES, key="pr_class")
    fig, ax = plt.subplots(figsize=(10, 6))
    pr_data = coco["pr_curves"]
    for cls_name, data in pr_data.items():
        if pr_class != "All" and cls_name != pr_class:
            continue
        prec = np.array(data["precision"])
        rec = np.array(data["recall"])
        valid = prec > -1
        if valid.any():
            ax.plot(rec[valid], prec[valid], label=f"{cls_name} (AP={np.mean(prec[valid]):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (IoU=0.50)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    _show_fig(fig)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.caption("Rows = ground truth class, columns = predicted class. 'background' = missed detections.")
    cm = eval_results["confusion_matrix"]
    cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, xticklabels=True, yticklabels=True)
    ax.set_title("Confusion Matrix (row-normalized)")
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    _show_fig(fig)

    # AP by size per class
    st.subheader("AP by Object Size")
    st.caption("COCO size thresholds: small < 32x32 px, medium < 96x96 px, large >= 96x96 px")
    pc = coco["per_class"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pc))
    w = 0.25
    ax.bar(x - w, pc["AP_small"], w, label="Small")
    ax.bar(x, pc["AP_medium"], w, label="Medium")
    ax.bar(x + w, pc["AP_large"], w, label="Large")
    ax.set_xticks(x)
    ax.set_xticklabels(pc["class"], rotation=45, ha="right")
    ax.set_ylabel("AP@50:95")
    ax.set_title("AP by Object Size per Class")
    ax.legend()
    plt.tight_layout()
    _show_fig(fig)

    # --- Failure Analysis section (same cache) ---
    st.subheader("Hardest Conditions (highest failure rate)")
    st.caption("Failure = image where model recall < 0.5")
    st.dataframe(eval_results["clusters"].head(15), hide_index=True)

    st.subheader("Phase 1 Data Features vs Model Recall")
    st.caption("Pearson/Spearman correlation between data characteristics and per-image recall")
    st.dataframe(eval_results["correlation_table"], hide_index=True)

    # GT vs Predictions browser
    st.subheader("GT vs Predictions Browser")
    preds_by_img = load_predictions()
    if not preds_by_img:
        st.info("No predictions loaded.")
        return

    failure_df = eval_results["failure_df"]
    browse_mode = st.selectbox("Browse", ["Worst Recall", "Random"], key="fail_browse")
    per_img = failure_df.sort_values("recall")
    if browse_mode == "Random":
        per_img = per_img.sample(min(20, len(per_img)))
    img_names = per_img.head(20)["image_name"].tolist()

    if img_names:
        selected = st.selectbox("Image", img_names, key="fail_img")
        row = per_img[per_img["image_name"] == selected].iloc[0]
        st.write(f"Precision: **{row['precision']:.3f}** | Recall: **{row['recall']:.3f}** | "
                 f"GT: {row['gt_count']} | Preds: {row['pred_count']} | {row['weather']}, {row['timeofday']}")

        img_path = IMAGE_DIRS["val"] / selected
        val_df = df[(df["image_name"] == selected) & (df["split"] == "val")]
        img_preds = preds_by_img.get(selected, [])

        if img_path.exists() and not val_df.empty:
            _show_fig(_render_gt_vs_pred(img_path, val_df, img_preds))



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main dashboard entry point."""
    st.set_page_config(page_title="BDD100K Analysis & Evaluation", layout="wide")
    st.title("BDD100K Object Detection — Analysis & Evaluation")

    df = load_data()
    metrics_df = load_metrics()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Overview", "Class Deep Dive", "Anomalies", "Safety-Critical Edge Cases",
         "Sample Browser", "Model Evaluation"]
    )

    with tab1:
        overview_tab(df)
    with tab2:
        class_deep_dive_tab(df)
    with tab3:
        anomalies_tab(df)
    with tab4:
        safety_critical_tab(df, metrics_df)
    with tab5:
        sample_browser_tab(df)
    with tab6:
        model_evaluation_tab(df)


if __name__ == "__main__":
    main()
