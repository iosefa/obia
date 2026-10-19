"""Create georeferenced objects with scikit-image segmentation algorithms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from geopandas import GeoDataFrame
from rasterio.features import shapes
from shapely.geometry import shape
from skimage import segmentation
from skimage.draw import polygon2mask
from skimage.measure import label

if TYPE_CHECKING:
    from obia.handlers.geotif import Image


SEGMENTATION_METHODS = (
    "slic", "quickshift", "felzenszwalb", "watershed", "random_walker",
    "chan_vese", "morphological_chan_vese",
    "morphological_geodesic_active_contour", "active_contour", "flood",
)
_SINGLE_BAND_METHODS = {
    "watershed", "chan_vese", "morphological_chan_vese",
    "morphological_geodesic_active_contour", "flood",
}
_BINARY_METHODS = {
    "chan_vese", "morphological_chan_vese",
    "morphological_geodesic_active_contour", "active_contour", "flood",
}
_CHANNEL_METHODS = {"slic", "quickshift", "felzenszwalb", "random_walker"}


def normalize_band(band):
    """Return a floating-point band scaled to [0, 1]; constant bands become zero."""
    values = np.asarray(band, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Segmentation bands must contain finite values inside the mask.")
    low, high = values.min(), values.max()
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _prepare_image(image, segmentation_bands, mask, normalize):
    data = np.asarray(image.img_data)
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("image.img_data must have shape (height, width, bands).")
    bands = list(range(data.shape[2])) if segmentation_bands is None else list(segmentation_bands)
    if not bands:
        raise ValueError("segmentation_bands must contain at least one band index.")
    for band in bands:
        if isinstance(band, (bool, np.bool_)) or not isinstance(band, (int, np.integer)):
            raise TypeError("segmentation_bands must contain integer band indexes.")
        if band < 0 or band >= data.shape[2]:
            raise IndexError(f"Band index {band} out of range; expected 0 to {data.shape[2] - 1}.")
    valid = np.ones(data.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != data.shape[:2]:
        raise ValueError("mask must have shape (height, width).")
    selected = np.array(data[:, :, bands], dtype=float, copy=True)
    if not np.isfinite(selected[valid]).all():
        raise ValueError("Segmentation bands must contain finite values inside the mask.")
    if normalize and valid.any():
        for band in range(selected.shape[2]):
            selected[:, :, band][valid] = normalize_band(selected[:, :, band][valid])
    selected[~valid] = 0
    return selected, valid


def _channel_image(data, method, params):
    """Keep channels last, or squeeze one band for grayscale algorithms."""
    default_axis = -1 if method == "quickshift" or data.shape[2] > 1 else None
    axis = params.setdefault("channel_axis", default_axis)
    if axis is None and method != "quickshift":
        if data.shape[2] != 1:
            raise ValueError("channel_axis=None requires selecting exactly one segmentation band.")
        return data[:, :, 0]
    if axis not in (-1, 2):
        raise ValueError("OBIA stores channels last; use channel_axis=-1 (or None for a single band).")
    if method == "quickshift":
        params.setdefault("convert2lab", data.shape[2] == 3)
    return data


def _check_markers(markers, image_shape, name):
    markers = np.asarray(markers)
    if markers.shape != image_shape or not np.issubdtype(markers.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array with shape (height, width).")
    if np.any(markers > np.iinfo(np.int64).max):
        raise ValueError(f"{name} values must fit in int64.")
    return markers.astype(np.int64, copy=True)


def create_labels(image: Image, segmentation_bands: Sequence[int] | None = None,
                  method: str = "slic", *, mask: np.ndarray | None = None,
                  normalize: bool = True, **kwargs: Any) -> np.ndarray:
    """Segment a raster into a two-dimensional integer object-label image.

    Args:
        image: OBIA Image with data shaped (height, width, bands).
        segmentation_bands: Zero-based bands to use; defaults to all bands.
            Watershed, Chan-Vese, morphological snakes, and flood require one band.
        method: One of ``SEGMENTATION_METHODS``; uses scikit-image's function names.
        mask: Boolean (height, width) array of included pixels. SLIC, watershed,
            and random walker respect it during segmentation. Other methods run
            on the prepared image and their results are clipped to the mask.
        normalize: Scale each selected band to [0, 1] using included pixels.
            Set False for physical-unit flood tolerances or preprocessed data.
        **kwargs: Parameters for the chosen scikit-image function. Random walker
            requires ``labels``, flood requires ``seed_point``, active contour
            requires a closed ``snake``, and morphological snakes require
            ``num_iter``. MorphGAC uses an inverse Gaussian gradient by default;
            ``gradient_kwargs`` configures it, or ``gimage`` supplies a precomputed
            (height, width) edge-stopping image. Diagnostic output flags
            (``return_tree``, ``extended_output``, ``return_full_prob``) are not
            supported because this function returns object labels.

    Returns:
        np.ndarray: An int32 (height, width) array. Each connected object has a unique
            positive ID; zero denotes background, watershed lines, or excluded
            pixels. Native algorithm/marker IDs are renumbered to match polygon
            ``segment_id`` values. Binary methods return foreground objects only.
            The source image and seed arrays are not modified.
    """
    if method not in SEGMENTATION_METHODS:
        raise ValueError(f"Unknown segmentation method {method!r}. Choose from: {', '.join(SEGMENTATION_METHODS)}.")
    params = dict(kwargs)
    for option in ("return_tree", "extended_output", "return_full_prob"):
        if params.get(option, False):
            raise ValueError(f"{option}=True does not return object labels; use scikit-image directly for diagnostic outputs.")

    if method == "random_walker":
        if "labels" not in params:
            raise ValueError("random_walker requires labels: positive seeds, 0 for unlabelled pixels, and negative inactive pixels.")
        params["labels"] = _check_markers(params["labels"], image.img_data.shape[:2], "labels")
        active = params["labels"] >= 0
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != active.shape:
                raise ValueError("mask must have shape (height, width).")
            active &= mask
        mask = active

    data, valid = _prepare_image(image, segmentation_bands, mask, normalize)
    if method in _SINGLE_BAND_METHODS and data.shape[2] != 1:
        raise ValueError(f"{method} requires one band; pass segmentation_bands=[band_index].")
    if not valid.any():
        return np.zeros(valid.shape, dtype=np.int32)
    if method in _CHANNEL_METHODS:
        source = _channel_image(data, method, params)
    elif method == "active_contour" and data.shape[2] > 1:
        source = data
    else:
        source = data[:, :, 0]

    if method == "slic" and mask is not None:
        params["mask"] = valid
    elif method == "watershed":
        params["mask"] = valid
        if params.get("markers") is not None and not np.isscalar(params["markers"]):
            params["markers"] = _check_markers(params["markers"], valid.shape, "markers")
    elif method == "random_walker":
        markers = params["labels"]
        markers[~valid] = -1
        if not np.any(markers > 0):
            raise ValueError("random_walker requires at least one positive seed inside the mask.")
        params["labels"] = markers
    elif method == "active_contour":
        if "snake" not in params:
            raise ValueError("active_contour requires snake: an (N, 2) array of (row, column) pixel coordinates.")
        snake = np.array(params["snake"], dtype=float, copy=True)
        if snake.ndim != 2 or snake.shape[1] != 2 or len(snake) < 3 or not np.isfinite(snake).all():
            raise ValueError("snake must contain at least three finite (row, column) coordinates.")
        if params.get("boundary_condition", "periodic") != "periodic":
            raise ValueError("Object polygons require boundary_condition='periodic'; use scikit-image directly for open snakes.")
        params["snake"] = snake
    elif method == "flood":
        if "seed_point" not in params:
            raise ValueError("flood requires seed_point=(row, column) in pixel coordinates.")
        seed = np.asarray(params["seed_point"])
        if (seed.shape != (2,) or not np.issubdtype(seed.dtype, np.integer)
                or any(v < 0 or v >= size for v, size in zip(seed, valid.shape))):
            raise ValueError("seed_point must contain two integer coordinates inside the image.")
        if not valid[tuple(seed)]:
            raise ValueError("seed_point must be inside the mask.")
    elif method == "morphological_geodesic_active_contour":
        gimage = params.pop("gimage", None)
        gradient_kwargs = params.pop("gradient_kwargs", {})
        if gimage is None:
            source = segmentation.inverse_gaussian_gradient(source, **gradient_kwargs)
        else:
            if gradient_kwargs:
                raise ValueError("Use either gimage or gradient_kwargs, not both.")
            source = np.array(gimage, dtype=float, copy=True)
            if source.shape != valid.shape or not np.isfinite(source).all():
                raise ValueError("gimage must be a finite (height, width) edge-stopping image.")
    if method.startswith("morphological_") and "num_iter" not in params:
        raise ValueError(f"{method} requires num_iter.")

    raw = getattr(segmentation, method)(source, **params)
    if method == "active_contour":
        raw = polygon2mask(valid.shape, raw)
    raw = np.asarray(raw)
    if raw.shape != valid.shape:
        raise ValueError(f"{method} did not return a two-dimensional segmentation.")
    included = valid.copy()
    if method in _BINARY_METHODS or method in {"watershed", "random_walker"}:
        included &= raw > 0
    # Each disconnected piece becomes its own object. Native zero labels from
    # SLIC/quickshift/Felzenszwalb are valid, unlike binary background.
    return label(np.where(included, raw, -1), background=-1, connectivity=1).astype(np.int32)


def labels_to_segments(labels, image):
    """Polygonize positive object labels using the raster's transform and CRS.

    ``labels`` must be a 2D integer array matching the raster, with background 0.
    Labels from :func:`create_labels` give one polygon per unique ``segment_id``.
    Custom labels with disconnected regions can produce multiple rows for an ID.
    """
    labels = np.asarray(labels)
    if labels.shape != image.img_data.shape[:2] or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be an integer (height, width) array matching the raster.")
    if labels.min() < 0 or labels.max() > np.iinfo(np.int32).max:
        raise ValueError("labels must be nonnegative int32 object IDs, with background 0.")
    records = [
        {"segment_id": int(value), "geometry": shape(geometry)}
        for geometry, value in shapes(labels.astype(np.int32), mask=labels > 0,
                                      transform=image.transform)
    ]
    return GeoDataFrame(records, columns=["segment_id", "geometry"], crs=image.crs)


def create_segments(image, segmentation_bands=None, method="slic", **kwargs):
    """Return georeferenced segment polygons for any ``SEGMENTATION_METHODS`` entry.

    Accepts the same parameters as :func:`create_labels`, including ``mask`` and
    ``normalize``. Zero/background pixels are omitted. The source raster is not
    modified; use :func:`create_labels` when only the label image is needed.
    """
    labels = create_labels(image, segmentation_bands, method, **kwargs)
    return labels_to_segments(labels, image)
