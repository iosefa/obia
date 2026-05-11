"""Point-cloud input normalization.

Path-based LAS/LAZ reading is optional and requires either PDAL or
pyforestscan. In-memory arrays and GeoDataFrames are supported without those
dependencies, which keeps the core package importable in lightweight
environments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


_X_NAMES = ("x", "X")
_Y_NAMES = ("y", "Y")
_Z_NAMES = ("z", "Z", "HeightAboveGround")
_INTENSITY_NAMES = ("intensity", "Intensity")


def read_pointcloud(
    pointcloud: str | Path | np.ndarray | pd.DataFrame | gpd.GeoDataFrame,
    *,
    crs: Any = None,
    reader: str = "auto",
) -> gpd.GeoDataFrame:
    """Return point-cloud data as a point GeoDataFrame.

    Parameters
    ----------
    pointcloud:
        LAS/LAZ path, structured NumPy array, ``(n, >=3)`` NumPy array,
        pandas DataFrame, or GeoDataFrame.
    crs:
        CRS assigned to in-memory point clouds when one is not already present.
        CRS alignment with segment polygons is checked later by
        ``add_pointcloud_features``.
    reader:
        Reader used for path inputs. ``"auto"`` tries PDAL first and then
        pyforestscan. ``"pdal"`` and ``"pyforestscan"`` force one backend.
    """
    if isinstance(pointcloud, gpd.GeoDataFrame):
        gdf = pointcloud.copy()
        if gdf.crs is None and crs is not None:
            gdf = gdf.set_crs(crs)
        return _normalize_point_geodataframe(gdf)

    if isinstance(pointcloud, pd.DataFrame):
        return _dataframe_to_points(pointcloud.copy(), crs=crs)

    if isinstance(pointcloud, np.ndarray):
        return _array_to_points(pointcloud, crs=crs)

    if isinstance(pointcloud, (str, Path)):
        return _read_pointcloud_path(Path(pointcloud), crs=crs, reader=reader)

    raise TypeError(
        "pointcloud must be a path, NumPy array, pandas DataFrame, or GeoDataFrame"
    )


def _read_pointcloud_path(path: Path, *, crs: Any = None, reader: str = "auto") -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if reader not in {"auto", "pdal", "pyforestscan"}:
        raise ValueError("reader must be 'auto', 'pdal', or 'pyforestscan'")

    errors = []

    if reader in {"auto", "pdal"}:
        try:
            return _read_with_pdal(path, crs=crs)
        except ImportError as exc:
            errors.append(str(exc))
            if reader == "pdal":
                raise

    if reader in {"auto", "pyforestscan"}:
        try:
            return _read_with_pyforestscan(path, crs=crs)
        except ImportError as exc:
            errors.append(str(exc))
            if reader == "pyforestscan":
                raise

    raise ImportError(
        "Reading point-cloud paths requires optional pointcloud dependencies. "
        "Install PDAL or pyforestscan, or pass an in-memory point cloud. "
        f"Backend errors: {'; '.join(errors)}"
    )


def _read_with_pdal(path: Path, *, crs: Any = None) -> gpd.GeoDataFrame:
    try:
        import pdal
    except ImportError as exc:
        raise ImportError("PDAL Python bindings are not installed.") from exc

    pipeline = pdal.Pipeline(
        json.dumps(
            [
                {
                    "type": "readers.las",
                    "filename": str(path),
                }
            ]
        )
    )
    pipeline.execute()
    if not pipeline.arrays:
        raise ValueError(f"No points were read from {path}")
    return _array_to_points(pipeline.arrays[0], crs=crs)


def _read_with_pyforestscan(path: Path, *, crs: Any = None) -> gpd.GeoDataFrame:
    try:
        from pyforestscan.handlers import read_lidar
    except ImportError as exc:
        raise ImportError("pyforestscan is not installed.") from exc

    if crs is None:
        raise ValueError("pyforestscan path reading requires an explicit crs")

    arrays = read_lidar(str(path), crs, hag=True)
    if not arrays:
        raise ValueError(f"No points were read from {path}")
    return _array_to_points(arrays[0], crs=crs)


def _array_to_points(array: np.ndarray, *, crs: Any = None) -> gpd.GeoDataFrame:
    if array.dtype.names:
        data = {}
        for source_names, target in [
            (_X_NAMES, "x"),
            (_Y_NAMES, "y"),
            (_Z_NAMES, "z"),
            (_INTENSITY_NAMES, "intensity"),
        ]:
            name = _first_present(array.dtype.names, source_names)
            if name is not None:
                data[target] = array[name]
        if "x" not in data or "y" not in data:
            raise ValueError("structured point cloud must contain x/y or X/Y fields")
        return _dataframe_to_points(pd.DataFrame(data), crs=crs)

    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("array point cloud must have shape (n, >=2)")

    columns = ["x", "y"]
    if array.shape[1] >= 3:
        columns.append("z")
    if array.shape[1] >= 4:
        columns.append("intensity")
    data = pd.DataFrame(array[:, : len(columns)], columns=columns)
    return _dataframe_to_points(data, crs=crs)


def _dataframe_to_points(df: pd.DataFrame, *, crs: Any = None) -> gpd.GeoDataFrame:
    x_col = _first_present(df.columns, _X_NAMES)
    y_col = _first_present(df.columns, _Y_NAMES)
    z_col = _first_present(df.columns, _Z_NAMES)
    intensity_col = _first_present(df.columns, _INTENSITY_NAMES)

    if x_col is None or y_col is None:
        raise ValueError("point cloud DataFrame must contain x/y or X/Y columns")

    out = pd.DataFrame(
        {
            "x": df[x_col].astype(float),
            "y": df[y_col].astype(float),
        }
    )
    if z_col is not None:
        out["z"] = df[z_col].astype(float)
    if intensity_col is not None:
        out["intensity"] = df[intensity_col].astype(float)

    return gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["x"], out["y"]), crs=crs)


def _normalize_point_geodataframe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.geometry.name is None:
        raise ValueError("GeoDataFrame point cloud must have an active geometry column")

    out = gdf.copy()
    geom = out.geometry
    out["x"] = geom.x.astype(float)
    out["y"] = geom.y.astype(float)

    if "z" not in out.columns and "Z" in out.columns:
        out["z"] = out["Z"].astype(float)
    elif "z" not in out.columns and geom.has_z.any():
        out["z"] = geom.apply(lambda point: point.z if point.has_z else np.nan).astype(float)

    if "intensity" not in out.columns and "Intensity" in out.columns:
        out["intensity"] = out["Intensity"].astype(float)

    return out


def _first_present(names, candidates) -> str | None:
    name_set = set(names)
    for candidate in candidates:
        if candidate in name_set:
            return candidate
    return None
