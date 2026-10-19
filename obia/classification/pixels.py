"""Sample labelled pixels and classify rasters using their loaded band values."""

from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, Sequence

from affine import Affine
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
import rasterio
from rasterio.transform import rowcol, xy
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix

from obia.classification.classify import _predict
from obia.classification.estimators import _classifier_pipeline
from obia.classification.splitting import ObjectSplit, _labels, split_objects
from obia.features._validation import band_indices

if TYPE_CHECKING:
    from obia.handlers.geotif import Image


def _raster_inputs(image, bands, mask):
    data = np.ma.asarray(image.img_data)
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("image.img_data must have nonempty shape (height, width, bands).")
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise TypeError("Raster bands must contain real numeric values.")
    selected = band_indices(bands, data.shape[2])
    if not selected:
        raise ValueError("bands must contain at least one band index.")
    if image.crs is None:
        raise ValueError("The raster must have a CRS.")
    CRS.from_user_input(image.crs)
    transform = image.transform
    if (not isinstance(transform, Affine) or not np.isfinite(tuple(transform)).all()
            or transform.is_degenerate):
        raise ValueError("The raster transform must be a finite, invertible Affine.")
    valid = np.ones(data.shape[:2], dtype=bool)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != valid.shape or mask.dtype != bool:
            raise ValueError("mask must be a boolean array with shape (height, width).")
        valid &= mask
    source_mask = getattr(image, "valid_mask", None)
    if source_mask is not None and np.shape(source_mask) != data.shape:
        raise ValueError("image.valid_mask must match image.img_data.")
    for band in selected:
        valid &= np.isfinite(data.data[:, :, band])
        valid &= ~np.ma.getmaskarray(data[:, :, band])
        if source_mask is not None:
            valid &= np.asarray(source_mask)[:, :, band].astype(bool, copy=False)
    return data.data, selected, valid


def sample_pixels(
    image: Image, points: gpd.GeoDataFrame, *, bands: Sequence[int] | None = None,
    label_column: str = "class", mask: np.ndarray | None = None,
    invalid_samples: Literal["raise", "drop"] = "raise",
) -> gpd.GeoDataFrame:
    """Read the pixel containing each labelled point, without interpolation.

    Args:
        image: Loaded raster with a CRS and affine transform.
        points: GeoDataFrame of finite, nonempty Point geometries with a CRS.
            Points are reprojected to the raster CRS when necessary.
        bands: Ordered, zero-based indices into the loaded image; None uses all.
        label_column: Column containing string or numeric class labels.
        mask: Boolean (height, width) inclusion mask, in addition to source masks.
        invalid_samples: Raise for points outside the raster or on excluded,
            NoData, or nonfinite pixels. "drop" explicitly discards these points.

    Returns:
        gpd.GeoDataFrame: One row per sampled pixel, with ``bN`` band values,
            ``feature_class``, ``pixel_row``, and ``pixel_col``. Geometry is the
            pixel centre in the raster CRS. Other point attributes and indices
            are retained; generated columns replace existing columns of those
            names. Repeated points in one pixel retain the first row if their
            labels agree; conflicting labels raise an error. Sampling counts
            are recorded in ``attrs['obia_sampling']``. Inputs are unchanged.
    """
    if invalid_samples not in ("raise", "drop"):
        raise ValueError("invalid_samples must be 'raise' or 'drop'.")
    data, selected, valid = _raster_inputs(image, bands, mask)
    if not isinstance(points, gpd.GeoDataFrame) or not points.columns.is_unique:
        raise TypeError("points must be a GeoDataFrame with unique column names.")
    if points.crs is None:
        raise ValueError("Sample points must have a CRS.")
    if label_column not in points or not len(points):
        raise ValueError(f"points must contain labelled rows and a {label_column!r} column.")
    geometry = points.geometry
    if (geometry.isna().any() or geometry.is_empty.any()
            or not geometry.geom_type.eq("Point").all() or not geometry.is_valid.all()):
        raise ValueError("Sample geometries must be finite, nonempty Points.")
    reserved = {"feature_class", "pixel_row", "pixel_col", *(f"b{b}" for b in selected)}
    if geometry.name in reserved or geometry.name == label_column:
        raise ValueError("The geometry column must be distinct from label and sample feature columns.")
    sampled = points.to_crs(image.crs).copy()
    if not np.isfinite(sampled.geometry.bounds.to_numpy()).all():
        raise ValueError("Sample geometries must have finite coordinates in the raster CRS.")
    rows, cols = rowcol(image.transform, sampled.geometry.x, sampled.geometry.y, op=np.floor)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    positions = np.flatnonzero(inside)
    inside[positions] &= valid[rows[positions].astype(int), cols[positions].astype(int)]
    n_invalid = int((~inside).sum())
    if n_invalid and invalid_samples == "raise":
        raise ValueError(
            f"{n_invalid} sample points fall outside the raster or on invalid pixels. "
            "Review them, or set invalid_samples='drop'."
        )
    sampled = sampled.iloc[np.flatnonzero(inside)].copy()
    if sampled.empty:
        raise ValueError("No valid labelled pixels remain.")
    rows, cols = rows[inside].astype(int), cols[inside].astype(int)
    sampled["feature_class"] = sampled[label_column].to_numpy(copy=True)
    _labels(sampled)
    sampled["pixel_row"], sampled["pixel_col"] = rows, cols
    for band in selected:
        sampled[f"b{band}"] = data[rows, cols, band]
    xs, ys = xy(image.transform, rows, cols)
    sampled = sampled.set_geometry(gpd.points_from_xy(xs, ys, crs=image.crs))
    counts = sampled.groupby(["pixel_row", "pixel_col"], sort=False).feature_class.nunique()
    if (counts > 1).any():
        raise ValueError("Conflicting class labels select the same raster pixel; resolve those labels first.")
    duplicates = sampled.duplicated(["pixel_row", "pixel_col"])
    sampled = sampled.loc[~duplicates].copy()
    sampled.attrs["obia_sampling"] = {
        "bands": selected, "label_column": label_column, "input_points": len(points),
        "invalid_points": n_invalid, "duplicate_points": int(duplicates.sum()),
        "sampled_pixels": len(sampled),
    }
    return sampled


@dataclass
class ClassifiedRaster:
    """Pixel predictions, georeferencing, and the fitted classifier.

    Attributes:
        classified: uint32 array (height, width). Codes 1..K follow ``classes``;
            zero is NoData. ``class_mapping`` recovers original string/numeric labels.
        crs: Raster coordinate reference system.
        transform: Raster affine transform.
        model: Fitted pipeline accepting columns named ``bN`` in band order.
        bands: Ordered zero-based indices used for learning and prediction.
        classes: Original class labels, in code and probability-column order.
        params: Estimator parameters.
        prediction_margin: Top-two probability gap, or NaN for invalid pixels,
            one-class models, or estimators without probabilities.
        probabilities: Optional float32 (height, width, classes) array; invalid
            pixels are NaN. None when not requested or unavailable.
        samples: Sampled training/holdout table, or None on prediction-only results.
        split: Sample partition used for fitting, or None on prediction-only results.
        confusion_matrix: Holdout matrix in class order, when requested.
        report: Holdout classification report, when requested.
    """

    classified: np.ndarray
    crs: Any
    transform: Affine
    model: BaseEstimator
    bands: tuple[int, ...]
    classes: np.ndarray
    params: dict
    prediction_margin: np.ndarray
    probabilities: np.ndarray | None = None
    samples: gpd.GeoDataFrame | None = None
    split: ObjectSplit | None = None
    confusion_matrix: np.ndarray | None = None
    report: str | None = None

    @property
    def class_mapping(self) -> dict:
        """Map positive raster codes to original class labels; 0 is NoData."""
        return dict(enumerate(self.classes.tolist(), start=1))

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Column order expected by the fitted model."""
        return tuple(f"b{band}" for band in self.bands)

    @property
    def valid_mask(self) -> np.ndarray:
        """Pixels with predictions."""
        return self.classified != 0

    def predict(
        self, image: Image, *, mask: np.ndarray | None = None,
        batch_size: int = 65536, return_probabilities: bool = False,
    ) -> ClassifiedRaster:
        """Predict another raster with the same band positions, without refitting.

        Args:
            image: Raster with matching band meanings, order, and value units.
                Its size, transform, and CRS may differ from the training raster.
            mask: Optional boolean (height, width) inclusion mask.
            batch_size: Maximum number of pixels processed per prediction batch.
            return_probabilities: Retain a full probability array when supported.

        Returns:
            ClassifiedRaster: Predictions on the new grid. Training samples,
                split, and holdout reports are not copied into this result.
        """
        return _predict_raster(image, self.model, self.bands, self.params,
                               mask, batch_size, return_probabilities)

    def write_geotiff(self, output_path: str) -> None:
        """Save uint32 class codes with CRS, transform, NoData=0, and label mapping.

        Args:
            output_path: Destination GeoTIFF. An existing file is replaced.

        The ``class_mapping`` dataset tag contains a JSON mapping of raster
        codes to original labels. The ``band_indices`` tag records model inputs.
        """
        with rasterio.open(
            output_path, "w", driver="GTiff", height=self.classified.shape[0],
            width=self.classified.shape[1], count=1, dtype="uint32", nodata=0,
            crs=self.crs, transform=self.transform, compress="deflate",
        ) as dst:
            dst.write(self.classified, 1)
            dst.set_band_description(1, "class_code")
            dst.update_tags(class_mapping=json.dumps(self.class_mapping),
                            band_indices=json.dumps(self.bands))


def _prediction_options(batch_size, return_probabilities):
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(return_probabilities, (bool, np.bool_)):
        raise ValueError("return_probabilities must be True or False.")


def _predict_raster(image, model, bands, params, mask, batch_size, return_probabilities):
    _prediction_options(batch_size, return_probabilities)
    data, selected, valid = _raster_inputs(image, bands, mask)
    classes = np.asarray(model.classes_)
    class_index = pd.Index(classes)
    shape = data.shape[:2]
    classified = np.zeros(shape, dtype=np.uint32)
    margin = np.full(shape, np.nan, dtype=np.float32)
    probabilities = (np.full((*shape, len(classes)), np.nan, dtype=np.float32)
                     if return_probabilities and hasattr(model, "predict_proba") else None)
    columns = [f"b{band}" for band in selected]
    for start in range(0, valid.size, batch_size):
        stop = min(start + batch_size, valid.size)
        positions = np.flatnonzero(valid.flat[start:stop]) + start
        if not len(positions):
            continue
        rows, cols = np.divmod(positions, shape[1])
        # Extract only this batch; never construct a feature table for the full raster.
        values = data[rows[:, None], cols[:, None], np.asarray(selected)]
        features = pd.DataFrame(values, columns=columns)
        labels, gaps, proba = _predict(model, features, features, None)
        codes = class_index.get_indexer(labels)
        if (codes < 0).any():
            raise ValueError("The classifier predicted a label outside model.classes_.")
        classified.flat[positions] = codes + 1
        margin.flat[positions] = gaps
        if probabilities is not None:
            probabilities[rows, cols, :] = proba.to_numpy()
    return ClassifiedRaster(classified, image.crs, image.transform, model, selected,
                            classes.copy(), params.copy(), margin, probabilities)


def classify_pixels(
    image: Image, points: gpd.GeoDataFrame, *, bands: Sequence[int] | None = None,
    label_column: str = "class", method: str | BaseEstimator = "rf",
    mask: np.ndarray | None = None, invalid_samples: Literal["raise", "drop"] = "raise",
    test_size: float | int | None = 0.2, compute_reports: bool = False,
    scale_features: bool | Literal["auto"] = "auto", split_random_state: int | None = 42,
    stratify: bool = True, split: ObjectSplit | None = None,
    batch_size: int = 65536, return_probabilities: bool = False, **kwargs: Any,
) -> ClassifiedRaster:
    """Learn from labelled point pixels and classify the raster directly.

    Args:
        image: Loaded raster. Source values are used without normalization.
        points: Labelled Point GeoDataFrame; see sample_pixels for sampling rules.
        bands: Ordered zero-based band indices; None selects all loaded bands.
        label_column: Point attribute containing string or numeric class labels.
        method: Any CLASSIFICATION_METHODS name or compatible classifier/pipeline.
        mask: Boolean inclusion mask applied to both sampling and prediction.
        invalid_samples: "raise" or "drop" for outside or invalid point pixels.
        test_size: Random sample holdout fraction/count; None fits all samples.
        compute_reports: Calculate holdout metrics; requires test samples.
        scale_features: "auto", True, or False, as in object classification.
            Preprocessing is fitted only on training samples.
        split_random_state: Random holdout seed, independent of model parameters.
        stratify: Preserve class coverage in the default random holdout.
        split: Reusable split created from sample_pixels with the same settings.
            Overrides test_size, split_random_state, and stratify. Spatial,
            grouped, explicit, and cross-validation partitions are supported.
        batch_size: Maximum number of pixels processed per prediction batch.
        return_probabilities: Retain the full probability array when available.
        **kwargs: Estimator constructor parameters or pipeline parameter overrides.

    Returns:
        ClassifiedRaster: Class codes, original labels, fitted model, sampling
            diagnostics, optional holdout reports, and raster georeferencing.
            Invalid pixels have code 0. Prediction batches limit temporary
            feature memory; the source and output rasters remain in memory.
    """
    _prediction_options(batch_size, return_probabilities)
    samples = sample_pixels(image, points, bands=bands, label_column=label_column,
                            mask=mask, invalid_samples=invalid_samples)
    selected = samples.attrs["obia_sampling"]["bands"]
    columns = [f"b{band}" for band in selected]
    y = _labels(samples)
    if split is None:
        split = split_objects(samples, test_size=test_size, random_state=split_random_state, stratify=stratify)
    elif not isinstance(split, ObjectSplit):
        raise TypeError("split must be an ObjectSplit created from sampled pixels.")
    split.validate_for(samples)
    if compute_reports and not len(split.test_indices):
        raise ValueError("compute_reports=True requires a holdout test subset.")
    if set(columns).intersection(split.feature_exclusions):
        raise ValueError("Pixel band columns cannot also be grouping columns.")
    model, params = _classifier_pipeline(method, scale_features, **kwargs)
    features = samples.loc[:, columns]
    model.fit(features.iloc[split.train_indices], y[split.train_indices])
    result = _predict_raster(image, model, selected, params, mask, batch_size, return_probabilities)
    result.samples, result.split = samples, split
    if compute_reports:
        observed = y[split.test_indices]
        predicted = model.predict(features.iloc[split.test_indices])
        result.confusion_matrix = confusion_matrix(observed, predicted, labels=result.classes)
        result.report = classification_report(observed, predicted, labels=result.classes, zero_division=0)
    return result
