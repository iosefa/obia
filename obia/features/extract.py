"""Calculate raster and geometric features for existing segment polygons."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from rasterio.windows import transform as window_transform

from ._validation import band_indices, choices
from .shape import SHAPE_FEATURES, shape_features
from .spectral import SPECTRAL_FEATURES, compute_index, summarize, validate_index, validate_statistics
from .texture import TextureConfig, measure_texture, texture_columns

if TYPE_CHECKING:
    from obia.handlers.geotif import Image


def _object_window(geometry, grid):
    if geometry.is_empty:
        return None
    try:
        window = geometry_window(grid, [geometry])
    except WindowError:
        return None
    rows, cols = window.toslices()
    transform = window_transform(window, grid.transform)
    inside = geometry_mask([geometry], out_shape=(rows.stop - rows.start, cols.stop - cols.start),
                           transform=transform, invert=True)
    return rows, cols, inside


def extract_features(
    segments: gpd.GeoDataFrame, image: "Image", *, bands: tuple[int, ...] | None = None,
    spectral: tuple[str, ...] = SPECTRAL_FEATURES, percentiles: tuple[float, ...] = (10, 90),
    indices: dict[str, tuple[int, ...]] | None = None,
    shape: tuple[str, ...] | bool = (), shape_units: str = "map",
    texture: TextureConfig | None = None, mask: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """Add selected features to a copy of an existing segment table.

    Args:
        segments: Polygon GeoDataFrame in the raster CRS. Its IDs, index,
            geometry, and existing attributes are preserved. Requested feature
            columns replace columns with the same names when recalculating.
        image: OBIA Image; calculations use its loaded (height, width, bands)
            values, including their selected order, rather than rereading files.
        bands: Zero-based band indices for spectral statistics and, by default,
            texture. None selects all loaded bands; () selects none.
        spectral: Names in ``SPECTRAL_FEATURES``. Use () to disable these summaries.
        percentiles: Additional percentiles in [0, 100], using linear interpolation.
            Use () to disable them. Applied to bands and calculated indices.
        indices: Index names mapped to ordered numeric band indices, e.g.
            {'ndvi': (3, 0)}. Indices are computed per pixel before summarization.
        shape: Names in ``SHAPE_FEATURES``, True for all, or False/() for none.
        shape_units: 'map' requires a projected CRS and uses its coordinate units;
            'pixel' uses the inverse raster transform, including rotation/shear.
        texture: Optional texture configuration. Ranges are fitted once per band
            over finite included image pixels, or supplied explicitly.
        mask: Optional boolean (height, width) mask. False pixels contribute to
            neither distributions nor texture. Geometry itself is unchanged.

    Returns:
        gpd.GeoDataFrame: Feature columns prefixed bN_, index_METHOD_bN_..., or
            shape_. Undefined measurements are NaN, valid counts are zero for
            objects without samples. Configuration and fitted texture ranges are
            recorded in attrs['obia_features'] (in-memory metadata).
    """
    data = np.ma.asarray(image.img_data)
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("image.img_data must have nonempty shape (height, width, bands).")
    if not isinstance(segments, gpd.GeoDataFrame):
        raise TypeError("segments must be a GeoDataFrame.")
    if (segments.crs is None) != (image.crs is None) or (
        segments.crs is not None and CRS.from_user_input(segments.crs) != CRS.from_user_input(image.crs)
    ):
        raise ValueError("segments and image must have the same CRS; reproject segments first.")
    for geometry in segments.geometry:
        if geometry is None or geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
            raise ValueError("segments must contain valid Polygon or MultiPolygon geometries.")
    selected = band_indices(bands, data.shape[2])
    spectral, percentiles = validate_statistics(spectral, percentiles)
    if shape is True:
        shape = SHAPE_FEATURES
    elif shape is False:
        shape = ()
    shape = choices(shape, SHAPE_FEATURES, "shape features")
    if shape_units not in {"map", "pixel"}:
        raise ValueError("shape_units must be 'map' or 'pixel'.")
    if shape and shape_units == "map" and (image.crs is None or not CRS.from_user_input(image.crs).is_projected):
        raise ValueError("Map-unit shape measurements require a projected CRS; use shape_units='pixel' instead.")
    included = np.ones(data.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if included.shape != data.shape[:2]:
        raise ValueError("mask must have shape (height, width).")
    image_mask = getattr(image, "valid_mask", None)
    if image_mask is not None and np.shape(image_mask) != data.shape:
        raise ValueError("image.valid_mask must match image.img_data.")
    if texture is not None and not isinstance(texture, TextureConfig):
        raise TypeError("texture must be a TextureConfig or None.")
    texture_bands = () if texture is None else band_indices(
        selected if texture.bands is None else texture.bands, data.shape[2])
    indices = {} if indices is None else dict(indices)
    indices = {name: validate_index(name, values, data.shape[2]) for name, values in indices.items()}

    cached_bands = {}

    def load_band(index):
        if index not in cached_bands:
            values = np.ma.asarray(data[:, :, index], dtype=float).filled(np.nan)
            valid = included & np.isfinite(values)
            if image_mask is not None:
                valid &= np.asarray(image_mask[:, :, index], dtype=bool)
            cached_bands[index] = np.where(valid, values, np.nan)
        return cached_bands[index]

    layers = {}
    if spectral or percentiles:
        layers.update({f"b{b}": load_band(b) for b in selected})
        for name, inputs in indices.items():
            values = np.stack([load_band(b) for b in inputs], axis=-1)
            prefix = f"index_{name}_" + "_".join(f"b{b}" for b in inputs)
            layers[prefix] = compute_index(values, name, tuple(range(len(inputs))))
    ranges = {}
    for band in texture_bands:
        values = load_band(band)
        valid = values[np.isfinite(values)]
        ranges[band] = texture.value_range or (
            (float(valid.min()), float(valid.max())) if valid.size else (0.0, 0.0))

    stats_columns = tuple(summarize([], 0, spectral, percentiles))
    columns = [f"{prefix}_{stat}" for prefix in layers for stat in stats_columns]
    columns += [f"b{b}_{name}" for b in texture_bands for name in texture_columns(texture)]
    columns += [f"shape_{name}" for name in shape]
    grid = SimpleNamespace(transform=image.transform, width=data.shape[1], height=data.shape[0])
    rows = []
    for geometry in segments.geometry:
        row = {}
        window = _object_window(geometry, grid) if layers or texture_bands else None
        if window is None:
            for prefix in layers:
                row.update({f"{prefix}_{name}": value for name, value in
                            summarize([], 0, spectral, percentiles).items()})
            for band in texture_bands:
                row.update({f"b{band}_{name}": np.nan for name in texture_columns(texture)})
        else:
            r, c, inside = window
            total = int(inside.sum())
            for prefix, values in layers.items():
                summary = summarize(values[r, c][inside], total, spectral, percentiles)
                row.update({f"{prefix}_{name}": value for name, value in summary.items()})
            for band in texture_bands:
                measured = measure_texture(load_band(band)[r, c], inside, texture, ranges[band])
                row.update({f"b{band}_{name}": value for name, value in measured.items()})
        if shape:
            measured = shape_features(geometry, shape, image.transform if shape_units == "pixel" else None)
            row.update({f"shape_{name}": value for name, value in measured.items()})
        rows.append(row)
    result = segments.copy()
    features = pd.DataFrame(rows, index=segments.index, columns=columns)
    # Assign together to avoid fragmenting tables with many bands and scales.
    result = gpd.GeoDataFrame(pd.concat([result.drop(columns=columns, errors="ignore"), features], axis=1),
                             geometry=segments.geometry.name, crs=segments.crs)
    result.attrs = dict(segments.attrs)
    result.attrs["obia_features"] = {
        "bands": selected, "spectral": spectral, "percentiles": percentiles,
        "indices": indices, "shape": shape, "shape_units": shape_units,
        "texture": asdict(texture) if texture is not None else None,
        "texture_ranges": ranges,
    }
    return result
