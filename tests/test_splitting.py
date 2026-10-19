"""Regression checks for spatial leakage, group isolation, and reusable folds."""

import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from numpy.testing import assert_allclose, assert_array_equal
from shapely.geometry import Polygon, box
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from obia.classification import classify, cross_validation_splits, split_objects


@pytest.fixture
def objects():
    site = np.repeat(np.arange(12), 6)
    kind = np.tile(np.arange(3), 24)
    return gpd.GeoDataFrame(
        {"segment_id": np.arange(72) + 100, "site": site,
         "feature_class": np.array(["forest", "soil", "water"])[kind],
         "b0_mean": kind * 3 + np.random.default_rng(7).normal(0, 0.2, 72)},
        geometry=[box((s % 4) * 100 + (i % 6) * 3 + 1, (s // 4) * 100 + 1,
                      (s % 4) * 100 + (i % 6) * 3 + 3, (s // 4) * 100 + 3)
                  for i, s in enumerate(site)],
        index=pd.Index([f"object-{i}" for i in range(72)], name="object"), crs="EPSG:32604",
    )


def assert_partition(plan, n):
    train, test, excluded = map(set, (plan.train_indices, plan.test_indices, plan.excluded_indices))
    assert not (train & test or train & excluded or test & excluded)
    assert train | test | excluded == set(range(n))
    assert len(plan.assignments) == n
    assert plan.class_counts.to_numpy().sum() == n


def test_random_holdout_matches_previous_default_and_returns_inspectable_copies(objects):
    before = objects.copy(deep=True)
    plan = split_objects(objects)
    expected_train, expected_test = train_test_split(
        np.arange(len(objects)), test_size=0.2, random_state=42, stratify=objects.feature_class,
    )
    assert_array_equal(plan.train_indices, expected_train)
    assert_array_equal(plan.test_indices, expected_test)
    assert_partition(plan, len(objects))
    assert plan.assignments.index.equals(objects.index)
    assert plan.assignments.crs == objects.crs
    assignments = plan.assignments
    assignments.loc[:, "subset"] = "modified"
    indices = plan.train_indices
    indices[:] = 0
    plan.metadata["method"] = "modified"
    assert_array_equal(plan.train_indices, expected_train)
    assert plan.metadata["method"] == "random"
    assert set(plan.assignments.subset) == {"train", "test"}
    assert_geodataframe_equal(objects, before)


@pytest.mark.parametrize("stratify", [True, False])
@pytest.mark.parametrize("method, options", [
    ("grouped", {"groups": "site"}), ("spatial", {"block_size": 100}),
])
def test_holdouts_keep_whole_groups_and_use_group_test_sizes(objects, method, options, stratify):
    plan = split_objects(objects, method=method, test_size=0.25, stratify=stratify, **options)
    train_sites = set(objects.iloc[plan.train_indices].site)
    test_sites = set(objects.iloc[plan.test_indices].site)
    assert len(test_sites) == 3
    assert not train_sites & test_sites
    assert len(plan.test_indices) == 18
    assert_partition(plan, len(objects))
    second = split_objects(objects, method=method, test_size=0.25, stratify=stratify, **options)
    assert_array_equal(plan.test_indices, second.test_indices)
    counted = split_objects(objects, method=method, test_size=2, stratify=stratify, **options)
    assert len(set(objects.iloc[counted.test_indices].site)) == 2


def test_group_balance_uses_object_class_counts_in_mixed_groups(objects):
    # Strongly unequal, mixed-class groups exercise approximate stratification.
    mixed = objects.drop(objects.index[[0, 1, 6, 12, 18, 24, 25, 26]])
    plan = split_objects(mixed, method="grouped", groups="site", test_size=0.3)
    assert (plan.class_counts[["train", "test"]] > 0).all().all()
    assert not set(mixed.iloc[plan.train_indices].site) & set(mixed.iloc[plan.test_indices].site)


def test_group_series_and_tuple_labels_are_positional_and_validated(objects):
    groups = objects.site.map(lambda value: ("site", value))
    plan = split_objects(objects, method="grouped", groups=groups)
    assert not set(groups.iloc[plan.train_indices]) & set(groups.iloc[plan.test_indices])
    with pytest.raises(ValueError, match="same index"):
        split_objects(objects, method="grouped", groups=groups.iloc[::-1])


def test_spatial_grid_uses_fixed_origin_negative_coordinates_and_whole_objects():
    objects = gpd.GeoDataFrame({"feature_class": ["a"] * 4},
                              geometry=[box(-11, 1, -9, 2), box(-1, 1, 1, 2),
                                        box(9, 1, 11, 2), box(19, 1, 21, 2)], crs=32604)
    plan = split_objects(objects, method="spatial", block_size=10, test_size=2)
    assert_array_equal(plan.assignments.block_x, [-1, 0, 1, 2])
    shifted = split_objects(objects, method="spatial", block_size=10, block_origin=(5, 0), test_size=2)
    assert_array_equal(shifted.assignments.block_x, [-2, -1, 0, 1])
    assert_geodataframe_equal(plan.assignments[["geometry"]], objects[["geometry"]])


def test_buffers_measure_geometry_edges_not_centroids_and_include_distance_boundary():
    objects = gpd.GeoDataFrame({"feature_class": ["a"] * 4},
                              geometry=[box(0, 0, 100, 1), box(101, 0, 102, 1),
                                        box(0, 30, 1, 31), box(0, 10, 1, 11)], crs=32604)
    # The long training polygon's centroid is distant, but its edge is 1 m away.
    plan = split_objects(objects, method="explicit", train_indices=[0, 2, 3], test_indices=[1],
                         buffer_distance=1)
    assert_array_equal(plan.excluded_indices, [0])
    assert_array_equal(plan.train_indices, [2, 3])
    for geometry in objects.geometry.iloc[plan.train_indices]:
        assert objects.geometry.iloc[plan.test_indices].distance(geometry).min() > 1
    assert_partition(plan, len(objects))


def test_buffer_removes_entire_training_groups_and_preserves_test_rows():
    objects = gpd.GeoDataFrame({"feature_class": ["a"] * 4, "site": [1, 1, 2, 3]},
                              geometry=[box(0, 0, 1, 1), box(50, 0, 51, 1),
                                        box(2, 0, 3, 1), box(100, 0, 101, 1)], crs=32604)
    plan = split_objects(objects, method="explicit", groups="site", train_indices=[0, 1, 3],
                         test_indices=[2], buffer_distance=1)
    assert_array_equal(plan.excluded_indices, [0, 1])
    assert_array_equal(plan.train_indices, [3])
    assert_array_equal(plan.test_indices, [2])


def test_buffer_zero_is_disabled_and_positive_buffers_remove_touching_polygons():
    objects = gpd.GeoDataFrame({"feature_class": [1, 1, 1]},
                              geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(10, 0, 11, 1)], crs=32604)
    for buffer, expected in ((0, [0, 2]), (0.01, [2])):
        plan = split_objects(objects, method="explicit", train_indices=[0, 2], test_indices=[1],
                             buffer_distance=buffer)
        assert_array_equal(plan.train_indices, expected)


def test_explicit_subsets_support_masks_exclusions_and_duplicate_source_indices(objects):
    objects.index = [0] * len(objects)
    train = np.arange(len(objects)) < 36
    test = np.arange(len(objects)) >= 48
    plan = split_objects(objects, method="explicit", train_indices=train, test_indices=test)
    assert_array_equal(plan.excluded_indices, np.arange(36, 48))
    result = classify(objects, objects, method="logistic_regression", split=plan,
                      feature_columns=["b0_mean"], compute_reports=True)
    assert result.split is plan
    assert result.confusion_matrix.sum() == test.sum()
    assert result.classified.index.equals(objects.index)


def test_classify_reuses_only_training_rows_and_excludes_numeric_group_column(objects):
    plan = split_objects(objects, method="grouped", groups="site", test_size=0.25)
    altered = objects.copy()
    altered.iloc[plan.test_indices, altered.columns.get_loc("b0_mean")] += 100
    # Changing features is safe; row identities and group membership stay fixed.
    result = classify(altered, altered, method="logistic_regression", split=plan, compute_reports=True)
    assert result.feature_columns == ("b0_mean",)
    assert_allclose(result.model.named_steps["scaler"].mean_,
                    altered.iloc[plan.train_indices][["b0_mean"]].mean())
    expected = make_pipeline(StandardScaler(), LogisticRegression()).fit(
        altered.iloc[plan.train_indices][["b0_mean"]], altered.iloc[plan.train_indices].feature_class,
    )
    expected_cm = confusion_matrix(altered.iloc[plan.test_indices].feature_class,
                                   expected.predict(altered.iloc[plan.test_indices][["b0_mean"]]),
                                   labels=result.classes)
    assert_array_equal(result.confusion_matrix, expected_cm)
    with pytest.raises(ValueError, match="grouping columns"):
        classify(objects, objects, split=plan, feature_columns=["b0_mean", "site"])


def test_nested_grouped_search_uses_development_positions_and_fits_preprocessing_inside_cv(objects):
    outer = split_objects(objects, method="grouped", groups="site", test_size=0.25)
    development = objects.iloc[outer.train_indices].copy()
    folds = cross_validation_splits(development, method="grouped", groups="site", n_splits=3)
    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression()),
        {"logisticregression__C": [0.1, 1]},
        cv=[fold.as_sklearn() for fold in folds], scoring="balanced_accuracy",
    )
    result = classify(objects.drop(columns="site"), objects, method=search,
                      split=outer, compute_reports=True)
    fitted_search = result.model.named_steps["classifier"]
    assert not hasattr(search, "best_estimator_")
    assert result.feature_columns == ("b0_mean",)
    assert result.confusion_matrix.sum() == len(outer.test_indices)
    assert_allclose(fitted_search.best_estimator_.named_steps["standardscaler"].mean_,
                    development[["b0_mean"]].mean())
    assert_allclose(fitted_search.cv_results_["mean_test_score"], 1)


def test_excluded_objects_are_not_used_to_fit_preprocessing(objects):
    plan = split_objects(objects, method="explicit", train_indices=np.arange(36),
                         test_indices=np.arange(36, 54))
    objects.iloc[plan.excluded_indices, objects.columns.get_loc("b0_mean")] = 1e6
    result = classify(objects, objects, method="logistic_regression", split=plan,
                      feature_columns=["b0_mean"], compute_reports=True)
    assert_allclose(result.model.named_steps["scaler"].mean_, objects.iloc[:36][["b0_mean"]].mean())
    assert result.confusion_matrix.sum() == 18
    assert len(result.classified) == len(objects)


def test_xgboost_reuses_grouped_split_with_original_string_classes(objects):
    pytest.importorskip("xgboost")
    plan = split_objects(objects, method="grouped", groups="site")
    result = classify(objects, objects, method="xgboost", split=plan, compute_reports=True,
                      n_estimators=5, max_depth=2, n_jobs=1, random_state=42)
    assert result.split is plan
    assert result.feature_columns == ("b0_mean",)
    assert set(result.classes) == set(objects.feature_class)
    assert result.confusion_matrix.sum() == len(plan.test_indices)


@pytest.mark.parametrize("change", ["order", "index", "labels", "ids", "geometry", "crs", "groups"])
def test_split_rejects_different_source_objects(objects, change):
    plan = split_objects(objects, method="grouped", groups="site")
    changed = objects.copy()
    if change == "order":
        changed = changed.iloc[::-1]
    elif change == "index":
        changed = changed.reset_index(drop=True)
    elif change == "labels":
        changed.iloc[0, changed.columns.get_loc("feature_class")] = "soil"
    elif change == "ids":
        changed.segment_id += 1
    elif change == "geometry":
        changed.geometry = changed.geometry.translate(1, 0)
    elif change == "crs":
        changed = changed.set_crs(32605, allow_override=True)
    else:
        changed.site += 1
    with pytest.raises(ValueError, match="does not match"):
        classify(changed, changed, method="dummy", split=plan)


def test_split_serialization_and_full_training_return_assignments(objects):
    plan = pickle.loads(pickle.dumps(split_objects(objects, test_size=None)))
    result = classify(objects, objects, method="dummy", split=plan)
    assert result.split.class_counts.train.sum() == len(objects)
    assert not len(result.split.test_indices)
    assert result.split.class_counts.test.sum() == 0
    with pytest.raises(ValueError, match="holdout"):
        classify(objects, objects, method="dummy", split=plan, compute_reports=True)


@pytest.mark.parametrize("method, options", [
    ("random", {}), ("grouped", {"groups": "site"}), ("spatial", {"block_size": 100}),
])
@pytest.mark.parametrize("stratify", [True, False])
def test_cross_validation_tests_every_object_once_and_accepts_sklearn_cv(objects, method, options, stratify):
    folds = cross_validation_splits(objects, method=method, n_splits=3, stratify=stratify, **options)
    assert_array_equal(np.sort(np.concatenate([fold.test_indices for fold in folds])), np.arange(len(objects)))
    for fold in folds:
        assert_partition(fold, len(objects))
        if method != "random":
            assert not set(objects.iloc[fold.train_indices].site) & set(objects.iloc[fold.test_indices].site)
    scores = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()),
                             objects[["b0_mean"]], objects.feature_class,
                             cv=[fold.as_sklearn() for fold in folds])
    assert_allclose(scores, 1)


def test_buffered_cross_validation_retains_test_coverage():
    objects = gpd.GeoDataFrame({"feature_class": ["a"] * 30},
                              geometry=[box(i * 10, 0, i * 10 + 1, 1) for i in range(30)], crs=32604)
    folds = cross_validation_splits(objects, method="spatial", block_size=30, n_splits=5,
                                    buffer_distance=9, stratify=False)
    assert sum(len(fold.test_indices) for fold in folds) == len(objects)
    assert any(len(fold.excluded_indices) for fold in folds)
    for fold in folds:
        for geometry in objects.geometry.iloc[fold.test_indices]:
            assert (objects.geometry.iloc[fold.train_indices].distance(geometry) > 9).all()
        assert_partition(fold, len(objects))


@pytest.mark.parametrize("options, message", [
    ({"method": "unknown"}, "method"),
    ({"method": "grouped"}, "requires groups"),
    ({"method": "grouped", "groups": "missing"}, "existing"),
    ({"method": "grouped", "groups": "feature_class"}, "non-label"),
    ({"method": "grouped", "groups": [1]}, "one hashable"),
    ({"method": "grouped", "groups": [None] * 72}, "cannot be missing"),
    ({"method": "grouped", "groups": [[1]] * 72}, "hashable"),
    ({"method": "grouped", "groups": "site", "n_candidates": 0}, "positive integer"),
    ({"method": "random", "groups": "site"}, "method='grouped'"),
    ({"method": "random", "block_size": 100}, "only used"),
    ({"method": "spatial", "block_size": 0}, "greater than zero"),
    ({"method": "spatial", "block_size": np.inf}, "finite"),
    ({"method": "spatial", "block_size": 100, "groups": "site"}, "not both"),
    ({"method": "spatial", "block_size": 100, "block_origin": (0,)}, "two finite"),
    ({"buffer_distance": -1}, "nonnegative"),
    ({"buffer_distance": np.nan}, "finite"),
    ({"test_size": True}, "test_size"),
    ({"stratify": "yes"}, "stratify"),
    ({"train_indices": [0]}, "method='explicit'"),
    ({"method": "explicit", "train_indices": [0]}, "test_indices"),
    ({"method": "explicit", "train_indices": [0, 0], "test_indices": [1]}, "distinct"),
    ({"method": "explicit", "train_indices": [-1], "test_indices": [1]}, "within"),
    ({"method": "explicit", "train_indices": [72], "test_indices": [1]}, "within"),
    ({"method": "explicit", "train_indices": [0.5], "test_indices": [1]}, "integer"),
    ({"method": "explicit", "train_indices": [True], "test_indices": [1]}, "one entry"),
    ({"method": "explicit", "train_indices": [0, 1, 2], "test_indices": [2, 3, 4]}, "overlap"),
    ({"method": "explicit", "groups": "site", "train_indices": [0, 1, 2], "test_indices": [3, 4, 5]}, "both training"),
])
def test_invalid_split_requests(objects, options, message):
    with pytest.raises(ValueError, match=message):
        split_objects(objects, **options)


@pytest.mark.parametrize("change", ["geographic", "no_crs", "plain", "empty", "missing", "invalid"])
def test_spatial_methods_and_buffers_validate_geometry_and_units(objects, change):
    if change == "geographic":
        objects = objects.to_crs(4326)
    elif change == "no_crs":
        objects = objects.set_crs(None, allow_override=True)
    elif change == "plain":
        objects = pd.DataFrame(objects.drop(columns="geometry"))
    elif change == "empty":
        objects.iloc[0, objects.columns.get_loc("geometry")] = Polygon()
    elif change == "missing":
        objects.iloc[0, objects.columns.get_loc("geometry")] = None
    else:
        objects.iloc[0, objects.columns.get_loc("geometry")] = Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])
    for options in ({"method": "spatial", "block_size": 100}, {"buffer_distance": 1}):
        with pytest.raises(ValueError, match="projected CRS|valid geometries"):
            split_objects(objects, **options)


def test_class_coverage_errors_are_explicit_and_do_not_split_groups(objects):
    objects.feature_class = np.where(objects.site == 0, "rare", "common")
    with pytest.raises(ValueError, match="No class-complete grouped split"):
        split_objects(objects, method="grouped", groups="site")
    with pytest.raises(ValueError, match="missing classes"):
        split_objects(objects, method="explicit", train_indices=np.arange(6, 72), test_indices=np.arange(6))
    plan = split_objects(objects, method="explicit", train_indices=np.arange(60), test_indices=np.arange(60, 72),
                         stratify=False)
    assert plan.class_counts.loc["rare", "test"] == 0
    with pytest.raises(ValueError, match="fold"):
        cross_validation_splits(objects, method="grouped", groups="site", n_splits=3)


def test_overlarge_buffer_fails_without_returning_empty_training_set(objects):
    with pytest.raises(ValueError, match="no training"):
        split_objects(objects, method="explicit", train_indices=np.arange(36), test_indices=np.arange(36, 72),
                       buffer_distance=1000)


@pytest.mark.parametrize("n_splits", [1, True, 2.5])
def test_invalid_cross_validation_counts(objects, n_splits):
    with pytest.raises(ValueError, match="n_splits"):
        cross_validation_splits(objects, n_splits=n_splits)
