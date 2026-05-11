"""Point-cloud feature calculations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_METRICS = ("height", "intensity", "density")
HEIGHT_PERCENTILES = (5, 25, 50, 75, 95)
INTENSITY_PERCENTILES = (5, 50, 95)


def calculate_pointcloud_features(
    points: pd.DataFrame,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    area: float | None = None,
    height_column: str = "z",
    intensity_column: str = "intensity",
) -> dict[str, float]:
    """Calculate point-cloud features for one segment.

    Parameters
    ----------
    points:
        Points inside one segment. The table may contain ``z`` and
        ``intensity`` columns.
    metrics:
        Feature groups to calculate. Supported values are ``"height"``,
        ``"intensity"``, and ``"density"``.
    area:
        Segment area in CRS units squared. Required for density.
    height_column:
        Column used for height metrics.
    intensity_column:
        Column used for intensity metrics.
    """
    metric_set = set(metrics)
    unknown = metric_set.difference(DEFAULT_METRICS)
    if unknown:
        raise ValueError(f"Unknown point-cloud metrics: {sorted(unknown)}")

    features: dict[str, float] = {"pc_point_count": int(len(points))}

    if "density" in metric_set:
        if area is None or area <= 0:
            features["pc_density"] = np.nan
        else:
            features["pc_density"] = float(len(points) / area)

    if "height" in metric_set:
        features.update(_numeric_series_features(points, height_column, "pc_z", HEIGHT_PERCENTILES))

    if "intensity" in metric_set:
        features.update(
            _numeric_series_features(points, intensity_column, "pc_intensity", INTENSITY_PERCENTILES)
        )

    return features


def empty_pointcloud_features(
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Return a feature dictionary filled with empty values."""
    return calculate_pointcloud_features(pd.DataFrame(), metrics=metrics, area=np.nan)


def _numeric_series_features(
    points: pd.DataFrame,
    column: str,
    prefix: str,
    percentiles: Iterable[int],
) -> dict[str, float]:
    values = _valid_numeric_values(points, column)
    features: dict[str, float] = {
        f"{prefix}_min": np.nan,
        f"{prefix}_max": np.nan,
        f"{prefix}_mean": np.nan,
        f"{prefix}_median": np.nan,
        f"{prefix}_std": np.nan,
    }
    for percentile in percentiles:
        features[f"{prefix}_p{percentile}"] = np.nan

    if values.size == 0:
        return features

    features[f"{prefix}_min"] = float(np.min(values))
    features[f"{prefix}_max"] = float(np.max(values))
    features[f"{prefix}_mean"] = float(np.mean(values))
    features[f"{prefix}_median"] = float(np.median(values))
    features[f"{prefix}_std"] = float(np.std(values))
    for percentile in percentiles:
        features[f"{prefix}_p{percentile}"] = float(np.percentile(values, percentile))
    return features


def _valid_numeric_values(points: pd.DataFrame, column: str) -> np.ndarray:
    if column not in points.columns:
        return np.array([], dtype=float)
    values = pd.to_numeric(points[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]
