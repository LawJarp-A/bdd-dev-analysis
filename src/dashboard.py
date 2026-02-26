"""Streamlit dashboard for BDD100K data analysis."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package is importable
# when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src import analysis  # noqa: E402
from src.compute_image_metrics import METRICS_PATH  # noqa: E402
from src.parser import DETECTION_CLASSES, IMAGE_DIRS, parse_labels  # noqa: E402


def _show_fig(fig: plt.Figure) -> None:
    """Display a matplotlib figure in Streamlit and close it to free memory."""
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the parsed dataset."""
    return parse_labels(split="all")


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    """Load pre-computed image quality metrics if available."""
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


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
        st.warning(
            "Statistically significant difference in class distribution between train and val splits."
        )
    else:
        st.success("No significant difference in class distribution between splits.")

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
    """Draw bounding boxes on an image.

    Args:
        img_path: Path to the image file.
        img_data: DataFrame rows for this image.
        highlight_indices: If provided, these DataFrame index values are drawn
            in bold red with labels. All other boxes are drawn in faded gray
            without labels. If None, all boxes are drawn normally with
            class-specific colors.
    """
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
        "Flags boxes that are unusually small or large **relative to their own class**. "
        "A 50 px\u00b2 traffic light is normal; a 50 px\u00b2 car is an outlier. "
        "These per-class outliers may indicate annotation errors, unusual viewing angles, "
        "or edge-case distances that the model will struggle with."
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
        f"**{len(tiny):,}** boxes ({len(tiny) / total * 100:.1f}% of all) occupy less than "
        f"1% of the image \u2014 most objects are at medium-to-far range. Models without "
        f"multi-scale feature extraction (e.g., FPN) will systematically miss these. "
        f"On the other end, **{len(huge):,}** boxes exceed 80% of image area (close-range "
        f"objects filling the frame)."
    )
    if not tiny.empty:
        st.dataframe(_anomaly_category_summary(tiny))
        _show_anomaly_sample(tiny, df, key="tiny_sample")

    # --- Extreme aspect ratios ---
    st.subheader("Extreme Aspect Ratios (< 0.1 or > 10)")
    extreme = analysis.extreme_aspect_ratios(df)
    st.write(
        f"**{len(extreme):,}** boxes have extreme width-to-height ratios. These typically "
        f"represent partially visible objects at frame edges (consistent with truncation) "
        f"or annotation artifacts. At {len(extreme) / total * 100:.2f}% of the dataset, "
        f"their training impact is negligible but they should be reviewed for quality."
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
        "These scenarios represent conditions where an object detection system is most "
        "likely to fail \u2014 and where failure has the highest real-world cost. Each case "
        "combines multiple difficulty factors (small size + poor visibility, occlusion + "
        "proximity to vehicles, etc.)."
    )

    _safety_case_section(
        title="Tiny Vulnerable Road Users at Night / Rain",
        explanation=(
            "Small pedestrians, riders, and cyclists in low-visibility conditions. "
            "These combine the two hardest detection challenges: small object size "
            "and degraded image quality. Missing a 20-pixel pedestrian at night is "
            "the canonical ADAS failure mode."
        ),
        case_df=analysis.tiny_vru_night_rain(df),
        full_df=df,
        key="safety_tiny_vru",
    )

    _safety_case_section(
        title="Occluded Pedestrians Near Cars",
        explanation=(
            "Pedestrians marked as occluded in images that also contain cars. "
            "This simulates people partially hidden behind parked or moving vehicles "
            "\u2014 a scenario responsible for a significant share of real-world "
            "pedestrian collisions."
        ),
        case_df=analysis.occluded_pedestrian_near_cars(df),
        full_df=df,
        key="safety_occluded_ped",
    )

    _safety_case_section(
        title="Crowded Night Intersections",
        explanation=(
            "City street scenes at night with 30+ annotated objects. High object "
            "density combined with low light stresses both the detector and NMS, "
            "increasing missed and merged detections."
        ),
        case_df=analysis.crowded_night_intersection(df),
        full_df=df,
        key="safety_crowded_night",
    )

    _safety_case_section(
        title="Truncated Pedestrians at Frame Edge",
        explanation=(
            "People partially cut off at the image boundary. These pedestrians are "
            "entering or leaving the camera\u2019s field of view and are easy for "
            "detectors to miss, yet critical for path planning and collision avoidance."
        ),
        case_df=analysis.truncated_person_edge(df),
        full_df=df,
        key="safety_truncated",
    )

    # Image-quality-based cases (require pre-computed metrics)
    if metrics_df is not None:
        _safety_case_section(
            title="Blurry Frames with Pedestrians",
            explanation=(
                "Images with low Laplacian variance (blur) that contain pedestrian "
                "annotations. Motion blur or camera shake degrades feature extraction "
                "and makes pedestrian detection unreliable in these safety-critical frames."
            ),
            case_df=analysis.blurry_with_pedestrians(df, metrics_df),
            full_df=df,
            key="safety_blurry",
        )

        _safety_case_section(
            title="Dark / Underexposed Frames with VRUs",
            explanation=(
                "Images with very low mean brightness containing pedestrians, riders, "
                "or cyclists. Low-light conditions reduce contrast and make vulnerable "
                "road users harder to distinguish from the background."
            ),
            case_df=analysis.dark_with_vru(df, metrics_df),
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
        "Under-represented environmental conditions. Models trained predominantly on "
        "clear/daytime data will have blind spots for these rare combinations. "
        "Targeted evaluation on these slices is essential."
    )
    combos = analysis.rare_condition_combos(df)
    st.dataframe(combos)

    # Blind spots callout
    st.subheader("Detection Blind Spots")
    st.info(
        "**Objects not covered by the 10 detection classes:**\n\n"
        "The BDD100K object detection task only labels bikes, buses, cars, motors, "
        "persons, riders, traffic lights, traffic signs, trains, and trucks. This means "
        "the following safety-relevant objects are **invisible** to any model trained on "
        "this dataset:\n\n"
        "- **Animals** (deer, dogs) \u2014 common collision cause in rural/suburban areas\n"
        "- **Road debris** (fallen cargo, tire fragments)\n"
        "- **Construction equipment** (cones, barriers, workers)\n"
        "- **Emergency vehicles** with non-standard profiles\n"
        "- **Wheelchairs, strollers, shopping carts** \u2014 non-standard VRUs\n"
        "- **Fallen pedestrians** \u2014 people lying on the road\n\n"
        "Any production deployment must augment this detector with additional models "
        "or a catch-all anomaly detection system to handle out-of-distribution objects."
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main dashboard entry point."""
    st.set_page_config(page_title="BDD100K Data Analysis", layout="wide")
    st.title("BDD100K Object Detection \u2014 Data Analysis")

    df = load_data()
    metrics_df = load_metrics()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Class Deep Dive", "Anomalies", "Safety-Critical Edge Cases", "Sample Browser"]
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


if __name__ == "__main__":
    main()
