"""Statistical analysis and plotting for BDD100K object detection data."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from src.parser import IMG_HEIGHT, IMG_WIDTH, VRU_CLASSES  # noqa: E402

IMG_AREA = IMG_WIDTH * IMG_HEIGHT


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["split", "category"]).size().unstack(fill_value=0)


def split_balance_chi2(df: pd.DataFrame) -> dict:
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df["category"], df["split"]))
    return {"chi2": chi2, "p_value": p, "dof": dof}


def cooccurrence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Co-occurrence via indicator dot product: C = ind.T @ ind."""
    ind = (
        df.groupby(["image_name", "category"]).size().unstack(fill_value=0) > 0
    ).astype(int)
    return ind.T @ ind


def objects_per_image(df: pd.DataFrame) -> pd.Series:
    return df.groupby("image_name").size()


def occlusion_truncation_rates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category").agg(
            occluded_pct=("occluded", "mean"),
            truncated_pct=("truncated", "mean"),
        )
        * 100
    ).round(2)


def extreme_aspect_ratios(
    df: pd.DataFrame, low: float = 0.1, high: float = 10.0
) -> pd.DataFrame:
    return df[(df["aspect_ratio"] < low) | (df["aspect_ratio"] > high)]


def per_class_outliers(
    df: pd.DataFrame, column: str = "area", k: float = 1.5
) -> pd.DataFrame:
    """Flag boxes outside per-class IQR fences (Q1 - k*IQR, Q3 + k*IQR).
    Adapts to each class's distribution so a small traffic light
    won't be flagged just because it's smaller than a car.
    """
    parts = []
    for _, grp in df.groupby("category"):
        q1, q3 = np.percentile(grp[column], [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        for label, mask in [("small", grp[column] < lo), ("large", grp[column] > hi)]:
            s = grp[mask].copy()
            s["outlier_type"] = label
            parts.append(s)
    return pd.concat(parts) if parts else pd.DataFrame()


def double_degraded(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["occluded"].astype(bool) & df["truncated"].astype(bool)]


def crowding_outliers(df: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """Images where a single class appears far more often than typical (> mean + k*std)."""
    counts = df.groupby(["image_name", "category"]).size().reset_index(name="count")
    per_class_stats = (
        counts.groupby("category")["count"].agg(["mean", "std"]).reset_index()
    )
    merged = counts.merge(per_class_stats, on="category")
    merged["threshold"] = merged["mean"] + k * merged["std"]
    flagged = merged[merged["count"] > merged["threshold"]]
    if flagged.empty:
        return pd.DataFrame()

    keys = set(zip(flagged["image_name"], flagged["category"]))
    mask = pd.Series(
        [(img, cat) in keys for img, cat in zip(df["image_name"], df["category"])],
        index=df.index,
    )
    result = df[mask].copy()
    count_map = dict(
        zip(zip(flagged["image_name"], flagged["category"]), flagged["count"])
    )
    result["class_count_in_image"] = [
        count_map.get((img, cat), 0)
        for img, cat in zip(result["image_name"], result["category"])
    ]
    return result


def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    class_distribution(df).T.plot(kind="bar", ax=ax)
    ax.set(title="Class Distribution by Split", ylabel="Count", xlabel="Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_cooccurrence(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cooccurrence_matrix(df), annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Class Co-occurrence Matrix")
    plt.tight_layout()
    return fig


def plot_bbox_area_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df, x="category", y="area", order=sorted(df["category"].unique()), ax=ax
    )
    ax.set_title("Bounding Box Area Distribution by Class")
    ax.set_ylabel("Area (px²)")
    ax.set_yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_objects_per_image(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = objects_per_image(df)
    ax.hist(counts, bins=50, edgecolor="black", alpha=0.7)
    ax.set(
        title="Objects per Image Distribution",
        xlabel="Number of Objects",
        ylabel="Number of Images",
    )
    ax.axvline(
        counts.mean(), color="red", linestyle="--", label=f"Mean: {counts.mean():.1f}"
    )
    ax.legend()
    plt.tight_layout()
    return fig


def plot_spatial_heatmap(df: pd.DataFrame, category: str | None = None) -> plt.Figure:
    subset = df[df["category"] == category] if category else df
    cx = (subset["x1"] + subset["x2"]) / 2
    cy = (subset["y1"] + subset["y2"]) / 2
    fig, ax = plt.subplots(figsize=(12, 7))
    h, _, _ = np.histogram2d(cx, cy, bins=50, range=[[0, IMG_WIDTH], [0, IMG_HEIGHT]])
    ax.imshow(
        h.T,
        origin="upper",
        extent=[0, IMG_WIDTH, 0, IMG_HEIGHT],
        aspect="auto",
        cmap="hot",
    )
    ax.set(
        title=f"Spatial Heatmap: {category or 'All Classes'}", xlabel="X", ylabel="Y"
    )
    plt.tight_layout()
    return fig


def plot_occlusion_truncation(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    occlusion_truncation_rates(df).plot(kind="bar", ax=ax)
    ax.set(title="Occlusion & Truncation Rates by Class", ylabel="Percentage (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# Safety-critical edge cases for autonomous driving


def tiny_vru_night_rain(
    df: pd.DataFrame, threshold: float = 0.0005, ego_lane_only: bool = False
) -> pd.DataFrame:
    adverse = (df["timeofday"] == "night") | (df["weather"] == "rainy")
    mask = (
        df["category"].isin(VRU_CLASSES) & (df["area"] < threshold * IMG_AREA) & adverse
    )
    if ego_lane_only and "in_ego_lane" in df.columns:
        mask = mask & df["in_ego_lane"].astype(bool)
    return df[mask]


def occluded_pedestrian_near_cars(
    df: pd.DataFrame, ego_lane_only: bool = False
) -> pd.DataFrame:
    car_imgs = set(df.loc[df["category"] == "car", "image_name"])
    adverse = (df["timeofday"] == "night") | (df["weather"] == "rainy")
    mask = (
        (df["category"] == "person")
        & df["occluded"].astype(bool)
        & (df["area"] < 0.001 * IMG_AREA)
        & df["image_name"].isin(car_imgs)
        & adverse
    )
    if ego_lane_only and "in_ego_lane" in df.columns:
        mask = mask & df["in_ego_lane"].astype(bool)
    return df[mask]


def crowded_night_intersection(df: pd.DataFrame, min_objects: int = 50) -> pd.DataFrame:
    night_city = df[(df["timeofday"] == "night") & (df["scene"] == "city street")]
    counts = night_city.groupby("image_name").size()
    return df[df["image_name"].isin(counts[counts >= min_objects].index)]


def truncated_person_edge(df: pd.DataFrame, edge_margin: int = 20) -> pd.DataFrame:
    at_edge = (
        (df["x1"] < edge_margin)
        | (df["x2"] > IMG_WIDTH - edge_margin)
        | (df["y1"] < edge_margin)
        | (df["y2"] > IMG_HEIGHT - edge_margin)
    )
    return df[(df["category"] == "person") & df["truncated"].astype(bool) & at_edge]


def blurry_with_pedestrians(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    blur_threshold: float = 15.0,
    ego_lane_only: bool = False,
) -> pd.DataFrame:
    flagged = set(
        metrics_df.loc[metrics_df["blur_score"] < blur_threshold, "image_name"]
    )
    cat_mask = df["category"] == "person"
    if ego_lane_only and "in_ego_lane" in df.columns:
        cat_mask = cat_mask & df["in_ego_lane"].astype(bool)
    target = flagged & set(df.loc[cat_mask, "image_name"])
    return df[df["image_name"].isin(target)]


def dark_with_vru(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    brightness_threshold: float = 20.0,
    ego_lane_only: bool = False,
) -> pd.DataFrame:
    flagged = set(
        metrics_df.loc[
            metrics_df["mean_brightness"] < brightness_threshold, "image_name"
        ]
    )
    cat_mask = df["category"].isin(VRU_CLASSES)
    if ego_lane_only and "in_ego_lane" in df.columns:
        cat_mask = cat_mask & df["in_ego_lane"].astype(bool)
    target = flagged & set(df.loc[cat_mask, "image_name"])
    return df[df["image_name"].isin(target)]


def rare_condition_combos(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    attrs = df.drop_duplicates(subset="image_name")[
        ["image_name", "weather", "timeofday"]
    ]
    return (
        attrs.groupby(["weather", "timeofday"])
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count")
        .head(top_n)
    )
