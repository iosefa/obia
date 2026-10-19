"""Pixel sampling, classifier compatibility, raster output, and spatial holdouts."""

import json

from affine import Affine
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point, box
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from obia.classification import (
    CLASSIFICATION_METHODS, ClassifiedRaster, classify_pixels, sample_pixels, split_objects,
)
from obia.handlers.geotif import Image


@pytest.fixture
def scene():
    values, classes = make_blobs(n_samples=90, centers=[[-5, -5], [0, 5], [5, -4]],
                                cluster_std=0.5, random_state=9)
    data = np.column_stack((values, np.arange(90))).reshape(9, 10, 3)
    transform = Affine(2, 0, 500000, 0, -3, 2200000)
    image = Image(data, "EPSG:32604", None, transform, None)
    rows, cols = np.indices(data.shape[:2])
    xs, ys = xy(transform, rows.ravel(), cols.ravel())
    points = gpd.GeoDataFrame(
        {"class": np.array(["forest", "water", "soil"])[classes],
         "site_id": np.repeat(np.arange(9), 10), "numeric_metadata": 999},
        geometry=gpd.points_from_xy(xs, ys), crs=image.crs,
        index=pd.Index([f"sample-{i}" for i in range(90)], name="sample"),
    )
    return image, points


def method_options(method):
    if method in {"rf", "extra_trees", "bagging", "gradient_boosting", "adaboost"}:
        return dict(n_estimators=5, random_state=2)
    if method.startswith("xgboost"):
        return dict(n_estimators=5, max_depth=2, n_jobs=1, random_state=2)
    if method == "hist_gradient_boosting":
        return dict(max_iter=8, min_samples_leaf=3, random_state=2)
    if method == "mlp":
        return dict(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=500, random_state=2)
    if method == "radius_neighbors":
        return dict(radius=20)
    if method == "gaussian_process":
        return dict(optimizer=None)
    if method in {"voting", "stacking"}:
        options = dict(estimators=[("tree", DecisionTreeClassifier()), ("lr", LogisticRegression())])
        if method == "stacking":
            options["cv"] = 2
        return options
    if method in {"one_vs_rest", "one_vs_one", "output_code"}:
        return dict(estimator=LogisticRegression())
    if method == "calibrated":
        return dict(estimator=LinearSVC(), cv=2)
    return {}


@pytest.mark.parametrize("method", CLASSIFICATION_METHODS)
@pytest.mark.parametrize("n_classes", [2, 3])
def test_every_named_classifier_predicts_pixels(scene, method, n_classes):
    image, points = scene
    if n_classes == 2:
        points = points[points["class"] != "soil"].copy()
    if method in {"multinomial_nb", "complement_nb", "categorical_nb", "bernoulli_nb"}:
        image.img_data = (image.img_data > 0).astype(np.int16)
    result = classify_pixels(image, points, bands=[1, 0], method=method, batch_size=23,
                             compute_reports=True, return_probabilities=True, **method_options(method))
    assert isinstance(result, ClassifiedRaster)
    assert result.classified.shape == image.img_data.shape[:2]
    assert result.classified.dtype == np.uint32
    assert result.bands == (1, 0) and result.feature_columns == ("b1", "b0")
    assert result.crs == image.crs and result.transform == image.transform
    assert result.valid_mask.all()
    assert result.confusion_matrix.shape == (n_classes, n_classes)
    assert result.confusion_matrix.sum() == len(points) // 5
    assert set(result.class_mapping.values()) == set(points["class"])
    features = pd.DataFrame(image.img_data.reshape(-1, 3)[:, [1, 0]], columns=result.feature_columns)
    assert_array_equal(result.classes[result.classified.ravel() - 1], result.model.predict(features))
    if hasattr(result.model, "predict_proba"):
        expected = result.model.predict_proba(features).reshape(9, 10, n_classes)
        assert_allclose(result.probabilities, expected, rtol=1e-6, atol=1e-7)
        top = np.sort(expected, axis=-1)[..., -2:]
        assert_allclose(result.prediction_margin, top[..., 1] - top[..., 0], atol=1e-7)
    else:
        assert result.probabilities is None
        assert np.isnan(result.prediction_margin).all()


@pytest.mark.parametrize("transform", [
    Affine(2, 0, 500000, 0, -3, 2200000),
    Affine.translation(500000, 2200000) * Affine.rotation(27) * Affine.shear(10) * Affine.scale(2, -3),
])
def test_sampling_uses_containing_pixel_preserves_order_and_reprojects(scene, transform):
    image, points = scene
    image.transform = transform
    rows, cols = np.array([0, 4, 8]), np.array([1, 3, 9])
    xs, ys = transform * (cols + 0.91, rows + 0.12)
    points = points.iloc[:3].set_geometry(gpd.points_from_xy(xs, ys, crs=image.crs)).to_crs(4326)
    points = points.rename_geometry("location").rename(columns={"class": "land_cover"})
    before = points.copy(deep=True)
    sampled = sample_pixels(image, points, bands=[2, 0], label_column="land_cover")
    assert_array_equal(sampled.pixel_row, rows)
    assert_array_equal(sampled.pixel_col, cols)
    assert_array_equal(sampled[["b2", "b0"]], image.img_data[rows[:, None], cols[:, None], [2, 0]])
    assert sampled.index.equals(points.index)
    assert sampled.geometry.name == "location"
    assert sampled.crs == rasterio.crs.CRS.from_user_input(image.crs)
    centre_x, centre_y = xy(transform, rows, cols)
    assert_allclose(sampled.geometry.x, centre_x)
    assert_allclose(sampled.geometry.y, centre_y)
    assert_array_equal(sampled.feature_class, points.land_cover)
    assert_array_equal(sampled.numeric_metadata, points.numeric_metadata)
    assert_geodataframe_equal(points, before)


def test_outside_points_and_pixel_edge_rules(scene):
    image, points = scene
    x, y = image.transform * (np.array([0, 10, 1, -0.01]), np.array([0, 1, 9, 1]))
    points = points.iloc[:4].set_geometry(gpd.points_from_xy(x, y, crs=image.crs))
    with pytest.raises(ValueError, match="3 sample points"):
        sample_pixels(image, points)
    sampled = sample_pixels(image, points, invalid_samples="drop")
    assert sampled[["pixel_row", "pixel_col"]].to_numpy().tolist() == [[0, 0]]
    assert sampled.attrs["obia_sampling"]["invalid_points"] == 3


def test_masks_nonfinite_values_and_unused_bands(scene):
    image, points = scene
    image.img_data = np.ma.array(image.img_data, mask=False)
    image.img_data.mask[0, 0, 0] = True
    image.img_data[0, 1, 1] = np.nan
    image.img_data[0, 2, 0] = np.inf
    image.img_data[0, 5, 2] = np.nan  # An unselected band must not exclude this pixel.
    image.valid_mask = np.ones(image.img_data.shape, dtype=bool)
    image.valid_mask[0, 3, 1] = False
    included = np.ones((9, 10), dtype=bool)
    included[0, 4] = False
    with pytest.raises(ValueError, match="5 sample points"):
        sample_pixels(image, points, bands=[0, 1], mask=included)
    result = classify_pixels(image, points, bands=[0, 1], mask=included, invalid_samples="drop",
                             method="decision_tree", test_size=None, return_probabilities=True)
    assert len(result.samples) == 85
    assert_array_equal(result.classified[0, :5], 0)
    assert np.isnan(result.prediction_margin[0, :5]).all()
    assert np.isnan(result.probabilities[0, :5]).all()
    assert result.valid_mask[0, 5]
    assert result.valid_mask.sum() == 85


def test_duplicate_pixels_are_deduplicated_and_conflicting_labels_rejected(scene):
    image, points = scene
    repeated = pd.concat([points, points.iloc[[0, 1]]])
    repeated.index = [7] * len(repeated)
    sampled = sample_pixels(image, repeated, bands=[0, 1])
    assert len(sampled) == 90
    assert sampled.attrs["obia_sampling"]["duplicate_points"] == 2
    assert sampled.index.tolist() == [7] * 90
    # A second location within the same pixel still refers to the same observation.
    repeated.iloc[-1, repeated.columns.get_loc("geometry")] = Point(*(image.transform * (1.9, 0.1)))
    repeated.iloc[-1, repeated.columns.get_loc("class")] = "conflict"
    with pytest.raises(ValueError, match="Conflicting class labels"):
        sample_pixels(image, repeated)


@pytest.mark.parametrize("method", ["random", "grouped", "spatial", "explicit"])
def test_reuses_point_partitions_and_fits_scaler_only_to_training(scene, method):
    image, points = scene
    sampled = sample_pixels(image, points, bands=[1, 0])
    options = {"random": {}, "grouped": {"groups": "site_id"},
               "spatial": {"block_size": 8},
               "explicit": {"train_indices": np.arange(60), "test_indices": np.arange(60, 90)}}[method]
    partition = split_objects(sampled, method=method, **options)
    result = classify_pixels(image, points, bands=[1, 0], method="logistic_regression",
                             split=partition, compute_reports=True)
    columns = list(result.feature_columns)
    expected = make_pipeline(StandardScaler(), LogisticRegression()).fit(
        sampled.iloc[partition.train_indices][columns], sampled.iloc[partition.train_indices].feature_class,
    )
    assert result.split is partition
    assert_allclose(result.model.named_steps["scaler"].mean_, expected.steps[0][1].mean_)
    heldout = sampled.iloc[partition.test_indices]
    assert_array_equal(result.confusion_matrix,
                       confusion_matrix(heldout.feature_class, expected.predict(heldout[columns]),
                                        labels=result.classes))
    with pytest.raises(ValueError, match="split does not match"):
        classify_pixels(image, points.iloc[::-1], bands=[1, 0], split=partition)


def test_custom_pipeline_is_cloned_and_model_can_be_reused_on_another_grid(scene, tmp_path):
    image, points = scene
    pipeline = make_pipeline(StandardScaler(), LinearSVC(random_state=5, max_iter=10000))
    result = classify_pixels(image, points, bands=[1, 0], method=pipeline,
                             linearsvc__C=0.4, compute_reports=True, return_probabilities=True)
    assert not hasattr(pipeline.steps[-1][1], "classes_")
    assert pipeline.steps[-1][1].C == 1
    assert result.probabilities is None and np.isnan(result.prediction_margin).all()
    new_image = Image(image.img_data[:4].copy(), "EPSG:26918", None,
                      Affine(1, 0.2, 300000, 0.1, -1, 4000000), None)
    predicted = result.predict(new_image, batch_size=7)
    assert predicted.model is result.model
    assert predicted.crs == new_image.crs and predicted.transform == new_image.transform
    assert predicted.samples is predicted.split is predicted.confusion_matrix is predicted.report is None
    assert_array_equal(predicted.classified, result.classified[:4])
    joblib.dump(result, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    assert_array_equal(loaded.predict(new_image).classified, predicted.classified)


@pytest.mark.parametrize("labels", [[0, 1000, 90000], ["forest", "water", "soil"]])
def test_geotiff_preserves_grid_nodata_and_original_labels(scene, labels, tmp_path):
    image, points = scene
    points["class"] = points["class"].map(dict(zip(["forest", "water", "soil"], labels)))
    result = classify_pixels(image, points, bands=[0, 1], method="xgboost", test_size=None,
                             n_estimators=5, max_depth=2, n_jobs=1)
    included = np.ones((9, 10), dtype=bool)
    included[0, 0] = False
    prediction = result.predict(image, mask=included)
    path = tmp_path / "classes.tif"
    prediction.write_geotiff(path)
    with rasterio.open(path) as src:
        assert src.shape == (9, 10) and src.count == 1 and src.dtypes == ("uint32",)
        assert src.nodata == 0 and src.crs == image.crs and src.transform == image.transform
        assert_array_equal(src.read(1), prediction.classified)
        assert_array_equal(src.read_masks(1) > 0, included)
        mapping = json.loads(src.tags()["class_mapping"])
        assert mapping == {str(k): v for k, v in result.class_mapping.items()}
        assert set(mapping.values()) == set(labels)
        assert json.loads(src.tags()["band_indices"]) == [0, 1]


def test_batch_sizes_are_respected_and_all_invalid_raster_skips_predict(scene, monkeypatch):
    image, points = scene
    result = classify_pixels(image, points, bands=[0, 1], method="rf", n_estimators=5,
                             random_state=42, return_probabilities=True)
    original = result.model.predict
    calls = []

    def predict(features):
        calls.append(len(features))
        assert len(features) <= 7
        return original(features)

    monkeypatch.setattr(result.model, "predict", predict)
    small = result.predict(image, batch_size=7, return_probabilities=True)
    assert len(calls) == 13
    assert_array_equal(small.classified, result.classified)
    assert_allclose(small.probabilities, result.probabilities)
    calls.clear()
    empty = result.predict(image, mask=np.zeros((9, 10), dtype=bool), return_probabilities=True)
    assert not calls
    assert_array_equal(empty.classified, 0)
    assert np.isnan(empty.prediction_margin).all() and np.isnan(empty.probabilities).all()


def test_more_than_255_classes_and_single_class(scene, tmp_path):
    image, points = scene
    points["class"] = "water"
    one = classify_pixels(image, points, method="dummy", test_size=None, return_probabilities=True)
    assert_array_equal(one.classified, 1)
    assert_array_equal(one.probabilities, 1)
    assert np.isnan(one.prediction_margin).all()
    image.img_data = np.arange(300).reshape(1, 300, 1)
    xs, ys = xy(image.transform, np.zeros(300), np.arange(300))
    points = gpd.GeoDataFrame({"class": np.arange(300) * 10},
                              geometry=gpd.points_from_xy(xs, ys), crs=image.crs)
    many = classify_pixels(image, points, method="decision_tree", test_size=None)
    assert_array_equal(many.classified, np.arange(1, 301).reshape(1, 300))
    many.write_geotiff(tmp_path / "many_classes.tif")
    with rasterio.open(tmp_path / "many_classes.tif") as src:
        assert src.read(1).max() == 300


@pytest.mark.parametrize("bands,error", [([], ValueError), ([0, 0], ValueError),
                                         ([True], TypeError), ([0.5], TypeError),
                                         ([-1], IndexError), ([3], IndexError)])
def test_invalid_band_indices(scene, bands, error):
    with pytest.raises(error):
        sample_pixels(*scene, bands=bands)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_size(scene, size):
    with pytest.raises(ValueError, match="batch_size"):
        classify_pixels(*scene, batch_size=size)


def test_rejects_invalid_points_metadata_and_predictions(scene):
    image, points = scene
    with pytest.raises(ValueError, match="CRS"):
        sample_pixels(image, points.set_crs(None, allow_override=True))
    with pytest.raises(ValueError, match="class"):
        sample_pixels(image, points.drop(columns="class"))
    bad = points.copy()
    bad.iloc[0, bad.columns.get_loc("geometry")] = box(0, 0, 1, 1)
    with pytest.raises(ValueError, match="Points"):
        sample_pixels(image, bad)
    with pytest.raises(ValueError, match="boolean"):
        sample_pixels(image, points, mask=np.ones((9, 10)))
    with pytest.raises(ValueError, match="boolean"):
        sample_pixels(image, points, mask=np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="No valid"):
        sample_pixels(image, points, mask=np.zeros((9, 10), dtype=bool), invalid_samples="drop")
    with pytest.raises(ValueError, match="invalid_samples"):
        sample_pixels(image, points, invalid_samples="ignore")
    with pytest.raises(ValueError, match="holdout"):
        classify_pixels(image, points, test_size=None, compute_reports=True)
    with pytest.raises(ValueError, match="Precomputed"):
        classify_pixels(image, points, method=SVC(kernel="precomputed"))
    with pytest.raises(ValueError, match="return_probabilities"):
        classify_pixels(image, points, return_probabilities=1)


def test_source_mask_shape_and_invalid_transform_are_rejected(scene):
    image, points = scene
    image.valid_mask = np.ones((9, 10), dtype=bool)
    with pytest.raises(ValueError, match="valid_mask"):
        sample_pixels(image, points)
    image.valid_mask = None
    image.transform = Affine(0, 0, 1, 0, 0, 1)
    with pytest.raises(ValueError, match="invertible"):
        sample_pixels(image, points)
