"""Train object classifiers and retain geospatial attributes in their predictions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils.multiclass import check_classification_targets

from obia.classification.estimators import (
    CLASSIFICATION_METHODS, _EncodedClassifier, _encode_xgboost,
    _is_xgboost_classifier, _requires_pairwise_input, _scale_automatically, create_classifier,
)
from obia.handlers.geotif import _write_geotiff


_RESERVED = {"feature_class", "segment_id", "predicted_class", "prediction_margin"}


class ClassifiedImage:
    """Classification output and its fitted, reusable preprocessing/model pipeline.

    Attributes:
        classified: Copy of the input segments with predictions and probability margins.
        model: Fitted scikit-learn pipeline; accepts the selected feature columns.
        feature_columns: Ordered feature names used for fitting and prediction.
        classes: Class labels in probability-column and confusion-matrix order.
        probabilities: Class probabilities indexed like the segments, or None.
        margin_kind: ``"probability"`` when probabilities exist, otherwise None.
        confusion_matrix: Held-out confusion matrix, or None.
        report: Held-out classification report, or None.
        params: Classifier parameters, excluding OBIA preprocessing options.
        shap_values: SHAP values for training features, or None.
    """

    def __init__(self, classified, confusion_matrix, report, shap_values, transform, crs,
                 params, *, model=None, feature_columns=(), classes=None, probabilities=None):
        self.classified = classified
        self.confusion_matrix = confusion_matrix
        self.report = report
        self.shap_values = shap_values
        self.transform = transform
        self.crs = crs
        self.params = params
        self.model = model
        self.feature_columns = tuple(feature_columns)
        self.classes = classes
        self.probabilities = probabilities
        self.margin_kind = "probability" if probabilities is not None else None

    def write_geotiff(self, output_path):
        """Write raster-valued classified data; polygon outputs must be rasterized first."""
        _write_geotiff(self.classified, output_path, self.crs, self.transform)


def _features(training, segments, columns):
    if not training.columns.is_unique or not segments.columns.is_unique:
        raise ValueError("Training and segment column names must be unique.")
    if columns is None:
        columns = [name for name in training.columns
                   if name not in _RESERVED and is_numeric_dtype(training[name].dtype)]
    else:
        if isinstance(columns, str):
            raise TypeError("feature_columns must be a sequence of column names.")
        columns = list(columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must select at least one feature, without duplicates.")
    if _RESERVED.intersection(columns):
        raise ValueError("Labels, segment IDs, and previous predictions cannot be used as features.")
    frames = []
    for name, table in (("training_classes", training), ("segments", segments)):
        missing = [column for column in columns if column not in table]
        if missing:
            raise ValueError(f"{name} is missing feature columns: {missing}.")
        nonnumeric = [column for column in columns
                      if not is_numeric_dtype(table[column].dtype) or is_complex_dtype(table[column].dtype)]
        if nonnumeric:
            raise ValueError(f"{name} has nonnumeric or complex feature columns: {nonnumeric}.")
        frame = table.loc[:, columns].copy()
        if np.isinf(frame.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{name} contains infinite feature values; replace them before classification.")
        # Keep NaNs for native missing-value support or user-supplied imputer pipelines.
        frames.append(frame)
    return columns, *frames


def _constraints(segments, acceptable):
    if acceptable is None:
        return None
    if not isinstance(acceptable, gpd.GeoDataFrame) or "acceptable_classes" not in acceptable:
        raise ValueError("acceptable_classes_gdf must be a GeoDataFrame with an acceptable_classes column.")
    if acceptable.crs != segments.crs:
        raise ValueError("acceptable_classes_gdf and segments must use the same CRS.")
    allowed = []
    for values in acceptable["acceptable_classes"]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise ValueError("Each acceptable_classes value must be a collection of class labels.")
        allowed.append(set(values))
    restrictions = []
    for geometry in segments.geometry:
        intersecting = np.flatnonzero(acceptable.intersects(geometry).to_numpy())
        restrictions.append(set.intersection(*(allowed[i] for i in intersecting))
                            if len(intersecting) else None)
    return restrictions


def _predict(model, features, segments, restrictions):
    classes = np.asarray(model.classes_)
    has_proba = hasattr(model, "predict_proba")
    if restrictions is not None and not has_proba:
        raise ValueError(
            "Acceptable-class constraints require predict_proba. Use a probabilistic "
            "classifier or wrap your estimator in CalibratedClassifierCV."
        )
    predictions = model.predict(features) if len(features) else np.array([], dtype=classes.dtype)
    if np.asarray(predictions).ndim != 1:
        raise ValueError("OBIA requires one predicted class per segment, not multioutput predictions.")
    predictions = np.asarray(predictions, dtype=object)
    margins = np.full(len(features), np.nan)
    probabilities = None
    if has_proba:
        values = model.predict_proba(features) if len(features) else np.empty((0, len(classes)))
        values = np.asarray(values)
        if values.shape != (len(features), len(classes)):
            raise ValueError("predict_proba must return one column per class and one row per segment.")
        probabilities = pd.DataFrame(values, index=segments.index, columns=classes)
        if len(classes) > 1:
            top = np.sort(values, axis=1)[:, -2:]
            margins = top[:, 1] - top[:, 0]
        if restrictions is not None:
            for position, allowed in enumerate(restrictions):
                if allowed is None:
                    continue
                candidates = np.flatnonzero([label in allowed for label in classes])
                margins[position] = np.nan
                if not len(candidates):
                    predictions[position] = pd.NA
                    continue
                scores = values[position, candidates]
                predictions[position] = classes[candidates[np.argmax(scores)]]
                if len(candidates) > 1:
                    top = np.sort(scores)[-2:]
                    margins[position] = top[1] - top[0]
    return predictions, margins, probabilities


def _explain(model, features, sample):
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP support requires the optional 'explain' dependencies. "
            "Install with `pip install 'obia[explain]'`."
        ) from exc
    background = shap.sample(features, min(500, len(features)), random_state=42) if sample else features
    estimator = model.steps[-1][1]
    transformed = model[:-1].transform(background) if len(model.steps) > 1 else background
    if isinstance(estimator, _EncodedClassifier):
        estimator = estimator.estimator_
    tree_types = (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier, ExtraTreeClassifier)
    is_xgb_tree = (_is_xgboost_classifier(estimator)
                   and estimator.get_params().get("booster") != "gblinear")
    if isinstance(estimator, tree_types) or is_xgb_tree:
        return shap.TreeExplainer(estimator).shap_values(transformed)
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "SHAP explanations for this estimator require predict_proba. "
            "Use CalibratedClassifierCV to add probabilities."
        )

    def predict_proba(values):
        return model.predict_proba(pd.DataFrame(values, columns=features.columns))

    return shap.KernelExplainer(predict_proba, background).shap_values(background)


def classify(segments: gpd.GeoDataFrame, training_classes: pd.DataFrame,
             acceptable_classes_gdf: gpd.GeoDataFrame | None = None,
             method: str | BaseEstimator = "rf", test_size: float | int | None = 0.2,
             compute_reports: bool = False, compute_shap: bool = False,
             sample_shap: bool = False, *, feature_columns: Sequence[str] | None = None,
             scale_features: bool | Literal["auto"] = "auto", split_random_state: int | None = 42,
             stratify: bool = True, **kwargs: Any) -> ClassifiedImage:
    """Fit a classifier on labelled objects and predict classes for segment polygons.

    Args:
        segments: Segment GeoDataFrame. The input is not modified.
        training_classes: Table with numeric features and a ``feature_class`` target.
            Labels may be strings or numeric class codes; missing labels are rejected.
        acceptable_classes_gdf: Optional polygons with lists of allowed class labels.
            Requires probabilities; overlapping polygons intersect their allowed sets.
        method: Name in ``CLASSIFICATION_METHODS`` or a scikit-learn-compatible
            classifier/pipeline instance, which is cloned before fitting.
        test_size: Holdout fraction or count; None trains on all labelled objects.
        compute_reports: Report held-out performance; requires a test split.
        compute_shap: Calculate SHAP explanations (requires the ``explain`` extra).
        sample_shap: Limit SHAP background and explained training rows to 500.
        feature_columns: Ordered feature names. Default selects numeric training columns,
            excluding IDs, labels, and previous prediction columns.
        scale_features: ``"auto"`` scales linear, SVM, neighbor, neural-network,
            discriminant, and Gaussian-process estimators. True always scales; False
            disables scaling. Custom pipelines and composite estimators own preprocessing.
            The scaler is fitted only on training rows and reused everywhere.
        split_random_state: Seed for the holdout split, independent of estimator parameters.
        stratify: Preserve class proportions in the holdout split (default True).
        **kwargs: Estimator parameters, including nested parameters for supplied pipelines.

    Returns:
        Object table, fitted model, feature order, probabilities, optional holdout
        reports, and optional SHAP values. Probability margins are NaN for
        estimators without probabilities or fewer than two allowed classes.
    """
    if not isinstance(segments, gpd.GeoDataFrame):
        raise TypeError("segments must be a GeoDataFrame.")
    if not training_classes.columns.is_unique or not segments.columns.is_unique:
        raise ValueError("Training and segment column names must be unique.")
    if "feature_class" not in training_classes or not len(training_classes):
        raise ValueError("training_classes must contain labelled rows and a feature_class column.")
    y = training_classes["feature_class"]
    if y.isna().any():
        raise ValueError("Training labels cannot be missing; remove unlabelled rows before classification.")
    string_labels = [isinstance(value, str) for value in y]
    if any(string_labels) and not all(string_labels):
        raise ValueError("Training labels must be all strings or all numeric class codes.")
    # Convert pandas nullable/categorical arrays to their natural NumPy label dtype.
    y = np.asarray(y.tolist())
    check_classification_targets(y)
    columns, x, x_pred = _features(training_classes, segments, feature_columns)
    restrictions = _constraints(segments, acceptable_classes_gdf)
    estimator = create_classifier(method, **kwargs)
    if _requires_pairwise_input(estimator):
        raise ValueError(
            "Precomputed kernels/distances are not object feature columns. Pass ordinary "
            "features, or a pipeline that builds its pairwise matrices from those features."
        )
    params = estimator.get_params()
    if scale_features != "auto" and not isinstance(scale_features, (bool, np.bool_)):
        raise ValueError("scale_features must be 'auto', True, or False.")
    scale = _scale_automatically(estimator) if scale_features == "auto" else scale_features
    steps = [("scaler", StandardScaler())] if scale else []
    model = Pipeline(steps + [("classifier", _encode_xgboost(estimator))])
    if test_size is None:
        if compute_reports:
            raise ValueError("compute_reports=True requires a holdout test_size.")
        x_train, y_train = x, y
    else:
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=split_random_state,
                stratify=y if stratify else None,
            )
        except ValueError as exc:
            raise ValueError(
                f"Cannot create the requested holdout split: {exc} "
                "Add labelled examples, adjust test_size, or use test_size=None to train on all rows."
            ) from exc
        if set(y_train) != set(y):
            raise ValueError("The training split is missing classes; use stratify=True or test_size=None.")
    model.fit(x_train, y_train)
    predictions, margins, probabilities = _predict(model, x_pred, segments, restrictions)
    classes = np.asarray(model.classes_)
    cm = report = None
    if compute_reports:
        predicted_test = model.predict(x_test)
        cm = confusion_matrix(y_test, predicted_test, labels=classes)
        report = classification_report(y_test, predicted_test, labels=classes, zero_division=0)
    shap_values = _explain(model, x_train, sample_shap) if compute_shap else None
    classified = segments.copy()
    label_dtype = pd.Series(classes).convert_dtypes().dtype
    classified["predicted_class"] = pd.array(predictions.tolist(), dtype=label_dtype)
    classified["prediction_margin"] = margins
    return ClassifiedImage(
        classified, cm, report, shap_values, None, segments.crs, params,
        model=model, feature_columns=columns, classes=classes, probabilities=probabilities,
    )
