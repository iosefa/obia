"""Join point-cloud feature groups onto segment objects."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from obia.pointcloud.features import DEFAULT_METRICS, calculate_pointcloud_features
from obia.pointcloud.io import read_pointcloud


def add_pointcloud_features(
    segments: gpd.GeoDataFrame,
    pointcloud: str | Path | np.ndarray | pd.DataFrame | gpd.GeoDataFrame,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    crs=None,
    reader: str = "auto",
    predicate: str = "within",
) -> gpd.GeoDataFrame:
    """Add point-cloud features to segment objects.

    Parameters
    ----------
    segments:
        Segment polygons. The output preserves the same rows and appends
        ``pc_*`` feature columns.
    pointcloud:
        LAS/LAZ path or in-memory point cloud. Path inputs require optional
        pointcloud dependencies.
    metrics:
        Feature groups to calculate: ``"height"``, ``"intensity"``, and
        ``"density"``.
    crs:
        CRS assigned to point clouds that do not already have one.
    reader:
        Backend for path inputs: ``"auto"``, ``"pdal"``, or ``"pyforestscan"``.
    predicate:
        Spatial join predicate used to attach points to polygons.
    """
    if segments.empty:
        return segments.copy()
    if segments.geometry.name is None:
        raise ValueError("segments must have an active geometry column")

    points = read_pointcloud(pointcloud, crs=crs or segments.crs, reader=reader)
    if points.empty:
        return _append_empty_features(segments, metrics)

    if segments.crs is not None:
        if points.crs is None:
            points = points.set_crs(segments.crs)
        elif points.crs != segments.crs:
            points = points.to_crs(segments.crs)

    segment_index_name = "__obia_segment_index"
    point_index_name = "__obia_point_index"

    segment_table = segments[[segments.geometry.name]].copy()
    segment_table[segment_index_name] = segments.index

    point_table = points.copy()
    point_table[point_index_name] = np.arange(len(point_table))

    joined = gpd.sjoin(
        point_table,
        segment_table,
        how="inner",
        predicate=predicate,
    )

    features_by_segment = {}
    for segment_index, group in joined.groupby(segment_index_name):
        points_in_segment = point_table.iloc[group[point_index_name].to_numpy()]
        area = float(segments.loc[segment_index].geometry.area)
        features_by_segment[segment_index] = calculate_pointcloud_features(
            points_in_segment,
            metrics=metrics,
            area=area,
        )

    empty_features = calculate_pointcloud_features(pd.DataFrame(), metrics=metrics, area=np.nan)
    output = segments.copy()
    for column in empty_features:
        output[column] = np.nan

    for segment_index, features in features_by_segment.items():
        for column, value in features.items():
            output.loc[segment_index, column] = value

    output["pc_point_count"] = output["pc_point_count"].fillna(0).astype(int)
    return output


def _append_empty_features(
    segments: gpd.GeoDataFrame,
    metrics: Iterable[str],
) -> gpd.GeoDataFrame:
    output = segments.copy()
    features = calculate_pointcloud_features(pd.DataFrame(), metrics=metrics, area=np.nan)
    for column, value in features.items():
        output[column] = value
    output["pc_point_count"] = output["pc_point_count"].fillna(0).astype(int)
    return output
