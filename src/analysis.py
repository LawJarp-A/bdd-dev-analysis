"""Statistical analysis and plotting for BDD100K object detection data."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


IMG_AREA = 1280 * 720


# ---------------------------------------------------------------------------
# Distribution Stats
# ---------------------------------------------------------------------------


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Class counts per split."""
    return df.groupby(["split", "category"]).size().unstack(fill_value=0)


def split_balance_chi2(df: pd.DataFrame) -> dict[str, float | int]:
    """Chi-squared test for class balance between train and val splits."""
    ct = pd.crosstab(df["category"], df["split"])
    chi2, p_value, dof, _ = stats.chi2_contingency(ct)
    return {"chi2": chi2, "p_value": p_value, "dof": dof}


def cooccurrence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Which classes appear together in the same image.

    Builds an indicator matrix (image x category), then computes
    co-occurrence via a dot product: C = indicator.T @ indicator.
    """
    indicator = df.groupby(["image_name", "category"]).size().unstack(fill_value=0)
    indicator = (indicator > 0).astype(int)
    matrix = indicator.T @ indicator
    return matrix


def objects_per_image(df: pd.DataFrame) -> pd.Series:
    """Number of detection objects per image."""
    return df.groupby("image_name").size()


def occlusion_truncation_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Occluded and truncated percentage per class."""
    return (
        df.groupby("category").agg(
            occluded_pct=("occluded", "mean"),
            truncated_pct=("truncated", "mean"),
        )
        * 100
    ).round(2)


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------


def tiny_boxes(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Boxes smaller than *threshold* fraction of image area."""
    return df[df["area"] < threshold * IMG_AREA]


def huge_boxes(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """Boxes larger than *threshold* fraction of image area."""
    return df[df["area"] > threshold * IMG_AREA]


def extreme_aspect_ratios(
    df: pd.DataFrame, low: float = 0.1, high: float = 10.0
) -> pd.DataFrame:
    """Boxes with extreme aspect ratios."""
    return df[(df["aspect_ratio"] < low) | (df["aspect_ratio"] > high)]


def per_class_outliers(
    df: pd.DataFrame, column: str = "area", low_pct: float = 5, high_pct: float = 95
) -> pd.DataFrame:
    """Flag boxes outside the per-class percentile range for a given column.

    Returns rows where the value is below the low_pct or above the high_pct
    percentile *within that row's class*. Adds an 'outlier_type' column.
    """
    results = []
    for category, group in df.groupby("category"):
        low_thresh = np.percentile(group[column], low_pct)
        high_thresh = np.percentile(group[column], high_pct)
        small = group[group[column] < low_thresh].copy()
        small["outlier_type"] = "small"
        large = group[group[column] > high_thresh].copy()
        large["outlier_type"] = "large"
        results.append(small)
        results.append(large)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Plot Functions
# ---------------------------------------------------------------------------


def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of class counts per split."""
    fig, ax = plt.subplots(figsize=(12, 6))
    dist = class_distribution(df)
    dist.T.plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution by Split")
    ax.set_ylabel("Count")
    ax.set_xlabel("Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_cooccurrence(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of class co-occurrence."""
    fig, ax = plt.subplots(figsize=(10, 8))
    matrix = cooccurrence_matrix(df)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Class Co-occurrence Matrix")
    plt.tight_layout()
    return fig


def plot_bbox_area_distribution(df: pd.DataFrame) -> plt.Figure:
    """Box plot of bbox areas per class."""
    fig, ax = plt.subplots(figsize=(12, 6))
    order = sorted(df["category"].unique())
    sns.boxplot(data=df, x="category", y="area", order=order, ax=ax)
    ax.set_title("Bounding Box Area Distribution by Class")
    ax.set_yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_objects_per_image(df: pd.DataFrame) -> plt.Figure:
    """Histogram of objects per image."""
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = objects_per_image(df)
    ax.hist(counts, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title("Objects per Image Distribution")
    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Number of Images")
    ax.axvline(
        counts.mean(), color="red", linestyle="--", label=f"Mean: {counts.mean():.1f}"
    )
    ax.legend()
    plt.tight_layout()
    return fig


def plot_spatial_heatmap(df: pd.DataFrame, category: str | None = None) -> plt.Figure:
    """Heatmap of bounding box centers."""
    subset = df[df["category"] == category] if category else df
    cx = (subset["x1"] + subset["x2"]) / 2
    cy = (subset["y1"] + subset["y2"]) / 2

    fig, ax = plt.subplots(figsize=(12, 7))
    heatmap, _xedges, _yedges = np.histogram2d(
        cx, cy, bins=50, range=[[0, 1280], [0, 720]]
    )
    ax.imshow(
        heatmap.T, origin="upper", extent=[0, 1280, 0, 720], aspect="auto", cmap="hot"
    )
    title = (
        f"Spatial Heatmap: {category}" if category else "Spatial Heatmap: All Classes"
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    return fig


def plot_occlusion_truncation(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of occlusion and truncation rates per class."""
    fig, ax = plt.subplots(figsize=(12, 6))
    rates = occlusion_truncation_rates(df)
    rates.plot(kind="bar", ax=ax)
    ax.set_title("Occlusion & Truncation Rates by Class")
    ax.set_ylabel("Percentage (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Safety-Critical Edge Cases
# ---------------------------------------------------------------------------

VRU_CLASSES = {"person", "rider", "bike"}


def tiny_vru_night_rain(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Tiny vulnerable road user boxes in night or rainy conditions.

    These are the hardest-to-detect, most dangerous-to-miss objects:
    small pedestrians/riders in poor visibility.
    """
    is_vru = df["category"].isin(VRU_CLASSES)
    is_tiny = df["area"] < threshold * IMG_AREA
    is_adverse = (df["timeofday"] == "night") | (df["weather"] == "rainy")
    mask = is_vru & is_tiny & is_adverse
    return df[mask]


def occluded_pedestrian_near_cars(df: pd.DataFrame) -> pd.DataFrame:
    """Occluded pedestrians in images that also contain cars.

    Simulates the real-world scenario of a pedestrian partially hidden
    by parked or moving vehicles — a leading cause of ADAS failures.
    """
    images_with_cars = set(df.loc[df["category"] == "car", "image_name"])
    is_person = df["category"] == "person"
    is_occluded = df["occluded"].astype(bool)
    in_car_image = df["image_name"].isin(images_with_cars)
    return df[is_person & is_occluded & in_car_image]


def crowded_night_intersection(
    df: pd.DataFrame, min_objects: int = 30
) -> pd.DataFrame:
    """Images with high object density at night in city streets.

    Dense urban scenes at night stress both detection and NMS, increasing
    the chance of missed or merged detections.
    """
    night_city = df[(df["timeofday"] == "night") & (df["scene"] == "city street")]
    counts = night_city.groupby("image_name").size()
    crowded_imgs = counts[counts >= min_objects].index
    return df[df["image_name"].isin(crowded_imgs)]


def truncated_person_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Truncated pedestrians — people entering/leaving the camera frame.

    Partially visible pedestrians at frame edges are easy for detectors
    to miss but critical for path planning.
    """
    return df[(df["category"] == "person") & df["truncated"].astype(bool)]


def _filter_by_metric_and_category(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    metric_column: str,
    metric_threshold: float,
    categories: set[str],
) -> pd.DataFrame:
    """Return rows from images that meet a metric threshold AND contain given categories.

    Shared logic for metric-based safety queries (blur, brightness, etc.).
    """
    flagged_imgs = set(
        metrics_df.loc[metrics_df[metric_column] < metric_threshold, "image_name"]
    )
    imgs_with_category = set(df.loc[df["category"].isin(categories), "image_name"])
    target_imgs = flagged_imgs & imgs_with_category
    return df[df["image_name"].isin(target_imgs)]


def blurry_with_pedestrians(
    df: pd.DataFrame, metrics_df: pd.DataFrame, blur_threshold: float = 50.0
) -> pd.DataFrame:
    """Images that are blurry and contain pedestrians.

    Motion blur or camera shake degrades feature extraction, making
    pedestrian detection unreliable in safety-critical frames.
    """
    return _filter_by_metric_and_category(
        df, metrics_df, "blur_score", blur_threshold, {"person"}
    )


def dark_with_vru(
    df: pd.DataFrame, metrics_df: pd.DataFrame, brightness_threshold: float = 60.0
) -> pd.DataFrame:
    """Dark/underexposed images containing vulnerable road users.

    Low-light conditions reduce contrast for pedestrians, riders, and
    cyclists who are already difficult to detect at distance.
    """
    return _filter_by_metric_and_category(
        df, metrics_df, "mean_brightness", brightness_threshold, VRU_CLASSES
    )


def rare_condition_combos(
    df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Rarest weather + time-of-day combinations by image count.

    Under-represented conditions (e.g., night+snow, dawn+foggy) are
    likely weak spots for any model trained on this dataset.
    """
    img_attrs = df.drop_duplicates(subset="image_name")[
        ["image_name", "weather", "timeofday"]
    ]
    combo_counts = (
        img_attrs.groupby(["weather", "timeofday"])
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count")
    )
    return combo_counts.head(top_n)
