"""Classifier integration and regressions for object tables and preprocessing."""

import builtins

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from numpy.testing import assert_allclose, assert_array_equal
from shapely.geometry import box
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from obia.classification import CLASSIFICATION_METHODS, classify, create_classifier


@pytest.fixture
def objects():
    x, y = make_blobs(n_samples=90, centers=[[-5, -5], [0, 5], [5, -4]],
                      cluster_std=0.5, random_state=9)
    segments = gpd.GeoDataFrame(
        {"segment_id": pd.array(np.arange(90) + 100, dtype="Int64"),
         "b0_mean": x[:, 0], "b1_mean": x[:, 1], "note": ["source"] * 90},
        geometry=[box(i * 2, 0, i * 2 + 1, 1) for i in range(90)], crs="EPSG:32604",
    )
    segments.index = pd.Index([f"object-{i * 7}" for i in range(90)], name="object")
    segments.attrs["source"] = "test-raster"
    training = segments.copy()
    training["feature_class"] = np.array(["forest", "water", "soil"])[y]
    return segments, training


def _method_options(method):
    if method in {"rf", "extra_trees", "bagging", "gradient_boosting", "adaboost"}:
        return dict(n_estimators=8, random_state=2)
    if method == "hist_gradient_boosting":
        return dict(max_iter=10, min_samples_leaf=3, random_state=2)
    if method == "mlp":
        return dict(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=500, random_state=2)
    if method == "radius_neighbors":
        return dict(radius=20)
    if method == "gaussian_process":
        return dict(optimizer=None)
    if method == "voting":
        return dict(estimators=[("tree", DecisionTreeClassifier()), ("lr", LogisticRegression())])
    if method == "stacking":
        return dict(estimators=[("tree", DecisionTreeClassifier()), ("lr", LogisticRegression())], cv=2)
    if method in {"one_vs_rest", "one_vs_one", "output_code"}:
        return dict(estimator=LogisticRegression())
    if method == "calibrated":
        return dict(estimator=LinearSVC(), cv=2)
    return {}


@pytest.mark.parametrize("method", [m for m in CLASSIFICATION_METHODS if not m.startswith("xgboost")])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_every_named_sklearn_classifier_trains_and_predicts(objects, method, n_classes):
    segments, training = objects
    if n_classes == 2:
        training = training[training.feature_class != "soil"].copy()
        segments = segments.loc[training.index].copy()
    if method in {"multinomial_nb", "complement_nb", "categorical_nb", "bernoulli_nb"}:
        # These distributions need nonnegative counts/categories or binary values.
        for table in (segments, training):
            for column in ("b0_mean", "b1_mean"):
                table[column] = (table[column] > 0).astype(int)
    result = classify(segments, training, method=method, compute_reports=True, **_method_options(method))
    assert result.confusion_matrix.sum() == len(training) // 5
    assert result.confusion_matrix.shape == (n_classes, n_classes)
    assert result.feature_columns == ("b0_mean", "b1_mean")
    assert set(result.classified.predicted_class) <= set(training.feature_class)
    assert_geodataframe_equal(result.classified[segments.columns], segments)
    assert result.classified.attrs == segments.attrs
    assert result.crs == segments.crs
    assert "predicted_class" not in segments
    assert_array_equal(result.model.predict(segments[list(result.feature_columns)]),
                       result.classified.predicted_class)
    if result.probabilities is not None:
        assert result.margin_kind == "probability"
        assert result.probabilities.index.equals(segments.index)
        assert_array_equal(result.probabilities.columns, result.classes)
        assert_allclose(result.probabilities.sum(axis=1), 1)
        top = np.sort(result.probabilities.to_numpy(), axis=1)[:, -2:]
        assert_allclose(result.classified.prediction_margin, top[:, 1] - top[:, 0])
    else:
        assert result.margin_kind is None
        assert result.classified.prediction_margin.isna().all()


def test_scaler_and_reports_use_training_partition_only(objects):
    segments, training = objects
    # A very different prediction distribution must never refit the scaler.
    segments["b0_mean"] += 1000
    result = classify(segments, training, method="logistic_regression", compute_reports=True)
    x_train, x_test, y_train, y_test = train_test_split(
        training[["b0_mean", "b1_mean"]], training.feature_class.to_numpy(),
        test_size=0.2, random_state=42, stratify=training.feature_class,
    )
    expected = make_pipeline(StandardScaler(), LogisticRegression()).fit(x_train, y_train)
    assert_allclose(result.model.named_steps["scaler"].mean_, x_train.mean())
    assert_allclose(result.probabilities, expected.predict_proba(segments[["b0_mean", "b1_mean"]]))
    assert_array_equal(result.confusion_matrix,
                       confusion_matrix(y_test, expected.predict(x_test), labels=result.classes))


def test_feature_order_metadata_old_predictions_and_custom_geometry(objects):
    segments, training = objects
    segments = segments.rename_geometry("outline")
    training = training.rename_geometry("outline")
    training["predicted_class"] = 999
    training["prediction_margin"] = 0.99
    first = classify(segments, training, method="rf", n_estimators=5, random_state=7)
    reordered = segments[["b1_mean", "note", "outline", "segment_id", "b0_mean"]].copy()
    reordered["unused_numeric"] = 7
    second = classify(reordered, training, method="rf", n_estimators=5, random_state=7)
    assert_array_equal(first.classified.predicted_class, second.classified.predicted_class)
    repeated = classify(first.classified, training, method="rf", n_estimators=5, random_state=7)
    assert_array_equal(first.classified.predicted_class, repeated.classified.predicted_class)
    assert first.classified.geometry.name == "outline"


def test_custom_pipeline_cloned_imputes_and_accepts_nested_parameters(objects):
    segments, training = objects
    for table in (segments, training):
        table.iloc[0, table.columns.get_loc("b0_mean")] = np.nan
    estimator = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                          ("lr", LogisticRegression())])
    result = classify(segments, training, method=estimator, lr__C=0.4, test_size=None)
    assert list(result.model.named_steps) == ["classifier"]
    fitted = result.model.named_steps["classifier"]
    assert fitted.named_steps["lr"].C == 0.4
    assert not hasattr(estimator.named_steps["lr"], "classes_")
    assert estimator.named_steps["lr"].C == 1
    assert_allclose(fitted.named_steps["impute"].statistics_,
                    training[["b0_mean", "b1_mean"]].mean())
    assert result.classified.predicted_class.notna().all()


@pytest.mark.parametrize("scale", [True, False])
def test_explicit_scaling_and_column_selection(objects, scale):
    segments, training = objects
    result = classify(segments, training, method="rf", n_estimators=5,
                      feature_columns=["b1_mean"], scale_features=scale, test_size=None)
    assert result.feature_columns == ("b1_mean",)
    assert ("scaler" in result.model.named_steps) is scale
    if scale:
        assert_allclose(result.model.named_steps["scaler"].mean_, training[["b1_mean"]].mean())


def test_numeric_labels_duplicate_indices_and_plain_training_table(objects):
    segments, training = objects
    training["feature_class"] = pd.array(training.feature_class.map({"forest": 10, "water": 30, "soil": 80}), dtype="Int64")
    segments.index = [7] * len(segments)
    training = pd.DataFrame(training.drop(columns=["geometry", "segment_id", "note"]))
    result = classify(segments, training, method="rf", n_estimators=5)
    assert set(result.classified.predicted_class) <= {10, 30, 80}
    assert result.classified.index.equals(segments.index)
    assert str(result.classified.predicted_class.dtype) == "Int64"


def test_one_class_and_empty_prediction_table(objects):
    segments, training = objects
    training.feature_class = "forest"
    result = classify(segments, training, method="dummy")
    assert set(result.classified.predicted_class) == {"forest"}
    assert result.classified.prediction_margin.isna().all()
    empty = classify(segments.iloc[:0], training, method="dummy")
    assert len(empty.classified) == 0
    assert empty.probabilities.shape == (0, 1)


def test_spatial_class_constraints_single_empty_unknown_and_overlapping(objects):
    segments, training = objects
    allowed = gpd.GeoDataFrame(
        {"acceptable_classes": [["forest"], ["unknown"], ["water", "soil"], ["forest", "soil"], []]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1),
                  box(4, 0, 5, 1), box(6, 0, 7, 1)], crs=segments.crs,
    )
    result = classify(segments, training, acceptable_classes_gdf=allowed, n_estimators=5)
    labels = result.classified.predicted_class
    assert labels.iloc[0] == "forest"
    assert pd.isna(labels.iloc[1])
    assert labels.iloc[2] == "soil"
    assert pd.isna(labels.iloc[3])
    assert result.classified.prediction_margin.iloc[:4].isna().all()
    assert result.classified.prediction_margin.iloc[4:].notna().all()


def test_constraint_margins_are_top_two_original_probabilities(objects):
    segments, training = objects
    allowed = gpd.GeoDataFrame({"acceptable_classes": [["forest", "water"]]},
                              geometry=[box(-1, -1, 1000, 2)], crs=segments.crs)
    result = classify(segments, training, method="gaussian_nb", acceptable_classes_gdf=allowed)
    assert_allclose(result.classified.prediction_margin,
                    (result.probabilities.forest - result.probabilities.water).abs())
    assert set(result.classified.predicted_class) <= {"forest", "water"}


def test_constraints_require_probabilities_and_matching_crs(objects):
    segments, training = objects
    allowed = gpd.GeoDataFrame({"acceptable_classes": [["forest"]]},
                              geometry=[box(0, 0, 1, 1)], crs=segments.crs)
    with pytest.raises(ValueError, match="predict_proba"):
        classify(segments, training, method="linear_svc", acceptable_classes_gdf=allowed)
    with pytest.raises(ValueError, match="same CRS"):
        classify(segments, training, acceptable_classes_gdf=allowed.to_crs("EPSG:4326"))
    allowed["acceptable_classes"] = "forest"
    with pytest.raises(ValueError, match="collection"):
        classify(segments, training, acceptable_classes_gdf=allowed)


@pytest.mark.parametrize("options, error", [
    ({"feature_columns": []}, "at least one"),
    ({"feature_columns": ["b0_mean", "b0_mean"]}, "duplicates"),
    ({"feature_columns": ["missing"]}, "missing feature"),
    ({"feature_columns": ["note"]}, "nonnumeric"),
    ({"feature_columns": ["feature_class"]}, "cannot be used"),
    ({"scale_features": "minmax"}, "scale_features"),
    ({"compute_reports": True, "test_size": None}, "holdout"),
    ({"method": "not_a_classifier"}, "Unknown classification"),
])
def test_invalid_requests_are_actionable(objects, options, error):
    segments, training = objects
    with pytest.raises(ValueError, match=error):
        classify(segments, training, **options)


def test_missing_labels_features_infinities_and_tiny_split(objects):
    segments, training = objects
    with pytest.raises(ValueError, match="missing feature"):
        classify(segments.drop(columns="b0_mean"), training)
    with pytest.raises(ValueError, match="infinite"):
        classify(segments.assign(b0_mean=np.inf), training)
    with pytest.raises(ValueError, match="cannot be missing"):
        classify(segments, training.assign(feature_class=pd.NA))
    tiny = training.groupby("feature_class").head(1)
    with pytest.raises(ValueError, match="test_size=None"):
        classify(segments, tiny)
    result = classify(segments, tiny, method="decision_tree", test_size=None)
    assert result.classified.predicted_class.notna().all()


@pytest.mark.parametrize("method", [LinearRegression(), object(), RandomForestClassifier])
def test_regressors_and_classes_are_not_classifier_instances(method):
    with pytest.raises(TypeError, match="classifier"):
        create_classifier(method)


@pytest.mark.parametrize("alias, canonical", [
    ("random_forest", "rf"), ("RandomForestClassifier", "rf"),
    ("SVC", "svc"), ("svm", "svc"), ("LogisticRegression", "logistic_regression"),
    ("extra-trees", "extra_trees"),
])
def test_named_aliases(alias, canonical):
    assert type(create_classifier(alias)) is type(create_classifier(canonical))


def test_optional_xgboost_import_is_lazy_and_error_is_actionable(monkeypatch, objects):
    original = builtins.__import__

    def no_xgboost(name, *args, **kwargs):
        if name.split(".")[0] == "xgboost":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_xgboost)
    classify(*objects, method="decision_tree")
    with pytest.raises(ImportError, match=r"obia\[xgboost\]"):
        create_classifier("xgboost")


@pytest.mark.parametrize("method", ["xgboost", "xgboost_rf", "xgb", "XGBClassifier"])
@pytest.mark.parametrize("numeric", [False, True])
def test_real_xgboost_multiclass_labels_and_probabilities(objects, method, numeric):
    pytest.importorskip("xgboost")
    segments, training = objects
    if numeric:
        training.feature_class = training.feature_class.map({"forest": 10, "water": 30, "soil": 80})
    result = classify(segments, training, method=method, n_estimators=5, max_depth=2,
                      n_jobs=1, random_state=2, compute_reports=True)
    assert set(result.classified.predicted_class) <= set(training.feature_class)
    assert_array_equal(result.classes, np.unique(training.feature_class))
    assert_allclose(result.probabilities.sum(axis=1), 1, atol=1e-6)
    assert result.confusion_matrix.sum() == 18
    assert_array_equal(result.model.predict(segments[list(result.feature_columns)]),
                       result.classified.predicted_class)
    assert np.mean(result.classified.predicted_class.to_numpy() == training.feature_class.to_numpy()) > 0.9


def test_xgboost_binary_custom_pipeline_and_missing_values(objects):
    xgb = pytest.importorskip("xgboost")
    segments, training = objects
    training = training[training.feature_class != "soil"].copy()
    training.iloc[0, training.columns.get_loc("b0_mean")] = np.nan
    segments.iloc[0, segments.columns.get_loc("b0_mean")] = np.nan
    estimator = make_pipeline(xgb.XGBClassifier(n_estimators=5, max_depth=2, n_jobs=1))
    result = classify(segments, training, method=estimator, test_size=None)
    assert set(result.classified.predicted_class) <= {"forest", "water"}
    assert result.probabilities.shape == (90, 2)
    assert not hasattr(estimator.steps[-1][1], "classes_")


@pytest.mark.parametrize("method", ["rf", "logistic_regression", "mlp", "xgboost"])
def test_real_shap_explanations(objects, method):
    pytest.importorskip("shap")
    if method == "xgboost":
        pytest.importorskip("xgboost")
    segments, training = objects
    training = training.groupby("feature_class").head(3)
    options = _method_options(method)
    if method == "xgboost":
        options = dict(n_estimators=3, n_jobs=1, max_depth=2)
    result = classify(segments.iloc[:3], training, method=method, test_size=None,
                      compute_shap=True, sample_shap=True, **options)
    assert np.asarray(result.shap_values).shape[0] == len(training)
    assert result.classified.predicted_class.notna().all()


def test_mixed_class_types_and_duplicate_columns_rejected(objects):
    segments, training = objects
    mixed = training.copy()
    mixed.feature_class = mixed.feature_class.astype(object)
    mixed.iloc[0, mixed.columns.get_loc("feature_class")] = 10
    with pytest.raises(ValueError, match="all strings or all numeric"):
        classify(segments, mixed)
    duplicated = pd.concat([training, training[["feature_class"]]], axis=1)
    with pytest.raises(ValueError, match="column names must be unique"):
        classify(segments, duplicated)


@pytest.mark.parametrize("numeric", [False, True])
def test_all_blocked_labels_keep_nullable_dtype_and_save_to_geopackage(objects, numeric, tmp_path):
    segments, training = objects
    if numeric:
        training.feature_class = training.feature_class.map({"forest": 10, "water": 30, "soil": 80})
    allowed = gpd.GeoDataFrame({"acceptable_classes": [[]]},
                              geometry=[box(-1, -1, 1000, 2)], crs=segments.crs)
    result = classify(segments, training, method="decision_tree", acceptable_classes_gdf=allowed)
    assert result.classified.predicted_class.isna().all()
    if numeric:
        assert str(result.classified.predicted_class.dtype) == "Int64"
    else:
        assert pd.api.types.is_string_dtype(result.classified.predicted_class.dtype)
    path = tmp_path / "classified.gpkg"
    result.classified.to_file(path, driver="GPKG")
    restored = gpd.read_file(path)
    assert len(restored) == len(segments)
    assert restored.predicted_class.isna().all()
    assert restored.crs == segments.crs


@pytest.mark.parametrize("method", ["calibrated", "bagging", "voting", "adaboost"])
def test_xgboost_inside_composite_models(objects, method):
    xgb = pytest.importorskip("xgboost")
    segments, training = objects
    estimator = xgb.XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1)
    if method == "voting":
        options = dict(estimators=[("xgb", estimator), ("tree", DecisionTreeClassifier())], voting="soft")
    elif method in {"bagging", "adaboost"}:
        options = dict(estimator=estimator, n_estimators=2)
    else:
        options = dict(estimator=estimator, cv=2)
    result = classify(segments, training, method=method, **options)
    assert set(result.classified.predicted_class) <= set(training.feature_class)
    assert not hasattr(estimator, "classes_")
    assert_allclose(result.probabilities.sum(axis=1), 1, atol=1e-6)


def test_categorical_training_labels(objects):
    segments, training = objects
    training.feature_class = training.feature_class.astype("category")
    result = classify(segments, training, method="decision_tree")
    assert set(result.classified.predicted_class) == set(training.feature_class)


def test_shap_is_optional_and_nonprobabilistic_model_gets_clear_error(objects, monkeypatch):
    segments, training = objects
    original = builtins.__import__

    def no_shap(name, *args, **kwargs):
        if name.split(".")[0] == "shap":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", no_shap)
        classify(segments, training, method="decision_tree")
        with pytest.raises(ImportError, match=r"obia\[explain\]"):
            classify(segments, training, method="decision_tree", compute_shap=True)
    pytest.importorskip("shap")
    with pytest.raises(ValueError, match="predict_proba"):
        classify(segments, training, method="linear_svc", compute_shap=True)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_custom_column_selector_receives_original_numeric_dtypes(objects, dtype):
    segments, training = objects
    for table in (segments, training):
        table["b0_mean"] = table.b0_mean.round().astype(dtype)
    selector = ColumnTransformer([
        ("integer", "passthrough", make_column_selector(dtype_include=np.integer)),
    ])
    result = classify(segments, training, method=make_pipeline(selector, LogisticRegression()), test_size=None)
    fitted = result.model.named_steps["classifier"]
    assert fitted.named_steps["columntransformer"].transformers_[0][2] == ["b0_mean"]
    assert np.mean(result.classified.predicted_class.to_numpy() == training.feature_class.to_numpy()) > 0.9


def test_complex_features_are_rejected_without_losing_imaginary_part(objects):
    segments, training = objects
    for table in (segments, training):
        table["b0_mean"] = table.b0_mean + 1j
    with pytest.raises(ValueError, match="complex feature"):
        classify(segments, training, method="decision_tree")


def test_xgboost_subclass_preserves_string_labels(objects):
    xgb = pytest.importorskip("xgboost")

    class CustomXGBClassifier(xgb.XGBClassifier):
        pass

    segments, training = objects
    estimator = CustomXGBClassifier(n_estimators=4, max_depth=2, n_jobs=1)
    result = classify(segments, training, method=estimator, test_size=None)
    assert_array_equal(result.classes, np.unique(training.feature_class))
    assert np.mean(result.classified.predicted_class.to_numpy() == training.feature_class.to_numpy()) > 0.9
    assert not hasattr(estimator, "classes_")


@pytest.mark.parametrize("nested", [False, True])
def test_xgboost_search_keeps_user_parameter_paths(objects, nested):
    xgb = pytest.importorskip("xgboost")
    segments, training = objects
    estimator = xgb.XGBClassifier(n_estimators=4, max_depth=2, n_jobs=1)
    if nested:
        estimator = Pipeline([("tree", estimator)])
    key = "tree__max_depth" if nested else "max_depth"
    search = GridSearchCV(estimator, {key: [1, 2]}, scoring="neg_log_loss", cv=2)
    result = classify(segments, training, method=search, test_size=None)
    fitted = result.model.named_steps["classifier"]
    assert fitted.best_params_[key] in {1, 2}
    assert fitted.best_estimator_.get_params()[key] == fitted.best_params_[key]
    assert np.mean(result.classified.predicted_class.to_numpy() == training.feature_class.to_numpy()) > 0.9
    assert not hasattr(search, "best_estimator_")


@pytest.mark.parametrize("selector", ["select_from_model", "rfe"])
def test_xgboost_importance_based_feature_selection(objects, selector):
    xgb = pytest.importorskip("xgboost")
    segments, training = objects
    estimator = xgb.XGBClassifier(n_estimators=4, max_depth=2, n_jobs=1)
    selection = (SelectFromModel(estimator, threshold=-np.inf) if selector == "select_from_model"
                 else RFE(estimator, n_features_to_select=1))
    pipeline = Pipeline([("selection", selection), ("classify", LogisticRegression())])
    result = classify(segments, training, method=pipeline, test_size=None)
    selected = result.model.named_steps["classifier"].named_steps["selection"]
    assert selected.get_support().sum() == (2 if selector == "select_from_model" else 1)
    assert set(result.classified.predicted_class) == set(training.feature_class)
    assert not hasattr(selection, "estimator_")


def test_xgboost_regressor_used_for_feature_selection_stays_a_regressor(objects):
    xgb = pytest.importorskip("xgboost")
    segments, training = objects
    training.feature_class = training.feature_class.map({"forest": 10, "water": 30, "soil": 80})
    pipeline = Pipeline([
        ("selection", SelectFromModel(xgb.XGBRegressor(n_estimators=4, n_jobs=1), threshold=-np.inf)),
        ("classify", LogisticRegression(max_iter=500)),
    ])
    result = classify(segments, training, method=pipeline, test_size=None)
    selected = result.model.named_steps["classifier"].named_steps["selection"]
    assert isinstance(selected.estimator_, xgb.XGBRegressor)
    assert set(result.classified.predicted_class) == {10, 30, 80}


def test_xgboost_nan_support_survives_bagging_wrapper(objects):
    xgb = pytest.importorskip("xgboost")
    segments, training = objects
    for table in (segments, training):
        table.iloc[0, table.columns.get_loc("b0_mean")] = np.nan
    result = classify(segments, training, method="bagging", n_estimators=2,
                      estimator=xgb.XGBClassifier(n_estimators=4, max_depth=2, n_jobs=1),
                      test_size=None, random_state=5)
    assert result.classified.predicted_class.notna().all()
    assert_allclose(result.probabilities.sum(axis=1), 1, atol=1e-6)


@pytest.mark.parametrize("method, options", [
    ("svc", {"kernel": "precomputed"}),
    ("nu_svc", {"kernel": "precomputed"}),
    ("knn", {"metric": "precomputed"}),
    ("radius_neighbors", {"metric": "precomputed"}),
])
def test_raw_precomputed_matrices_get_actionable_error(objects, method, options):
    with pytest.raises(ValueError, match="Precomputed kernels/distances"):
        classify(*objects, method=method, **options)


class _PairwiseKernel(TransformerMixin, BaseEstimator):
    """Build Gram/cross-kernel matrices inside the training pipeline."""

    def fit(self, X, y=None):
        self.reference_ = np.array(X)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.reference_)


def test_pipeline_can_build_its_own_pairwise_matrices(objects):
    segments, training = objects
    pipeline = make_pipeline(_PairwiseKernel(), SVC(kernel="precomputed"))
    result = classify(segments, training, method=pipeline, compute_reports=True)
    assert result.confusion_matrix.sum() == 18
    assert np.mean(result.classified.predicted_class.to_numpy() == training.feature_class.to_numpy()) > 0.9


def test_search_can_choose_xgboost_as_an_estimator(objects):
    xgb = pytest.importorskip("xgboost")
    segments, training = objects
    choices = [LogisticRegression(), xgb.XGBClassifier(n_estimators=4, max_depth=2, n_jobs=1)]
    pipeline = Pipeline([("model", LogisticRegression())])
    search = GridSearchCV(pipeline, {"model": choices}, scoring="neg_log_loss", cv=2, error_score="raise")
    result = classify(segments, training, method=search, test_size=None)
    fitted = result.model.named_steps["classifier"]
    assert np.isfinite(fitted.cv_results_["mean_test_score"]).all()
    assert set(result.classified.predicted_class) == set(training.feature_class)
    assert not hasattr(choices[1], "classes_")


def test_xgboost_feature_selector_inside_column_transformer(objects):
    xgb = pytest.importorskip("xgboost")
    segments, training = objects
    columns = ColumnTransformer([
        ("selected", SelectFromModel(xgb.XGBClassifier(n_estimators=4, n_jobs=1), threshold=-np.inf),
         ["b0_mean", "b1_mean"]),
    ])
    result = classify(segments, training, method=make_pipeline(columns, LogisticRegression()), test_size=None)
    assert set(result.classified.predicted_class) == set(training.feature_class)


@pytest.mark.parametrize("method, options", [
    ("svc", {"probability": True, "random_state": 3}),
    ("nu_svc", {"probability": True, "random_state": 3}),
    ("sgd", {"loss": "log_loss", "random_state": 3}),
    ("sgd", {"loss": "modified_huber", "random_state": 3}),
    ("voting", {"estimators": [("lr", LogisticRegression()), ("tree", DecisionTreeClassifier())], "voting": "soft"}),
])
def test_probability_enabled_variants_support_class_constraints(objects, method, options):
    segments, training = objects
    allowed = gpd.GeoDataFrame({"acceptable_classes": [["forest", "water"]]},
                              geometry=[box(-1, -1, 1000, 2)], crs=segments.crs)
    result = classify(segments, training, method=method, acceptable_classes_gdf=allowed, **options)
    assert result.margin_kind == "probability"
    assert_allclose(result.probabilities.sum(axis=1), 1)
    assert set(result.classified.predicted_class) <= {"forest", "water"}
    assert_allclose(result.classified.prediction_margin,
                    (result.probabilities.forest - result.probabilities.water).abs())


@pytest.mark.parametrize("name", ["FixedThresholdClassifier", "TunedThresholdClassifierCV"])
def test_additional_sklearn_threshold_classifiers(objects, name):
    from sklearn import model_selection
    if not hasattr(model_selection, name):
        pytest.skip("Threshold classifiers require scikit-learn 1.5 or later")
    segments, training = objects
    training = training[training.feature_class != "soil"].copy()
    training.feature_class = training.feature_class.map({"forest": 0, "water": 1})
    estimator = getattr(model_selection, name)(LogisticRegression())
    result = classify(segments, training, method=estimator, test_size=None)
    assert set(result.classified.predicted_class) <= {0, 1}
    assert not hasattr(estimator, "estimator_")


@pytest.mark.parametrize("name", ["LabelPropagation", "LabelSpreading"])
def test_additional_sklearn_classifiers_with_fully_labelled_data(objects, name):
    from sklearn import semi_supervised
    segments, training = objects
    training.feature_class = training.feature_class.map({"forest": 10, "water": 30, "soil": 80})
    estimator = getattr(semi_supervised, name)()
    result = classify(segments, training, method=estimator, test_size=None)
    assert set(result.classified.predicted_class) == {10, 30, 80}
    assert_allclose(result.probabilities.sum(axis=1), 1)


@pytest.mark.parametrize("method", ["rf", "svc", "xgboost"])
def test_fitted_model_can_be_saved_reloaded_and_retrained(objects, method, tmp_path):
    import joblib
    from sklearn.base import clone
    if method == "xgboost":
        pytest.importorskip("xgboost")
    segments, training = objects
    options = {"n_estimators": 4, "n_jobs": 1} if method in {"rf", "xgboost"} else {}
    result = classify(segments, training, method=method, **options)
    path = tmp_path / "classifier.joblib"
    joblib.dump(result.model, path)
    restored = joblib.load(path)
    features = segments[list(result.feature_columns)]
    assert_array_equal(restored.predict(features), result.classified.predicted_class)
    if result.probabilities is not None:
        assert_allclose(restored.predict_proba(features), result.probabilities)
    retrained = clone(restored).fit(training[list(result.feature_columns)], training.feature_class)
    assert set(retrained.predict(features)) == set(training.feature_class)


def test_frozen_classifier_and_parameter_overrides_preserve_original(objects):
    frozen = pytest.importorskip("sklearn.frozen")
    segments, training = objects
    features = training[["b0_mean", "b1_mean"]]
    fitted = LogisticRegression().fit(features, training.feature_class)
    original = frozen.FrozenEstimator(fitted)
    predictions = original.predict(features)
    result = classify(segments, training, method=original, test_size=None)
    assert_array_equal(result.classified.predicted_class, predictions)
    replacement = DecisionTreeClassifier().fit(features, training.feature_class)
    copied = create_classifier(original, estimator=replacement)
    assert original.estimator is fitted
    assert copied.estimator is replacement
    assert_array_equal(original.predict(features), predictions)


@pytest.mark.parametrize("calibrated", [False, True])
def test_frozen_xgboost_can_be_used_directly_or_calibrated(objects, calibrated):
    frozen = pytest.importorskip("sklearn.frozen")
    xgb = pytest.importorskip("xgboost")
    from sklearn.calibration import CalibratedClassifierCV
    segments, training = objects
    training.feature_class = training.feature_class.map({"forest": 0, "water": 1, "soil": 2})
    features = training[["b0_mean", "b1_mean"]]
    fitted = xgb.XGBClassifier(n_estimators=4, max_depth=2, n_jobs=1).fit(features, training.feature_class)
    original = frozen.FrozenEstimator(fitted)
    state_before = fitted.get_booster().save_raw()
    method = CalibratedClassifierCV(original) if calibrated else original
    result = classify(segments, training, method=method, test_size=None)
    assert set(result.classified.predicted_class) == {0, 1, 2}
    assert original.estimator is fitted
    assert fitted.get_booster().save_raw() == state_before
    assert_array_equal(original.predict(features), fitted.predict(features))


@pytest.mark.parametrize("method, extra", [
    ("extra_trees", {}), ("decision_tree", {}), ("extra_tree", {}),
    ("gradient_boosting", {}), ("hist_gradient_boosting", {}),
    ("gaussian_nb", {}), ("knn", {}), ("gaussian_process", {}),
    ("calibrated", {}), ("sgd", {"loss": "log_loss", "random_state": 3}),
    ("xgboost_rf", {"n_estimators": 3, "n_jobs": 1, "max_depth": 2}),
    ("xgboost", {"n_estimators": 3, "n_jobs": 1, "booster": "gblinear"}),
    ("xgboost", {"n_estimators": 3, "n_jobs": 1, "max_depth": 2, "booster": "dart"}),
])
def test_shap_across_other_probabilistic_families(objects, method, extra):
    pytest.importorskip("shap")
    if method.startswith("xgboost"):
        pytest.importorskip("xgboost")
    segments, training = objects
    training = training.groupby("feature_class").head(3)
    options = {**_method_options(method), **extra}
    result = classify(segments.iloc[:3], training, method=method, test_size=None,
                      compute_shap=True, sample_shap=True, **options)
    assert np.asarray(result.shap_values).shape[:2] == (9, 2)
    assert np.isfinite(np.asarray(result.shap_values)).all()
