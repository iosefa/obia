"""Numerical and geospatial contracts for raster object features."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.affinity import rotate
from shapely.geometry import Polygon, box
from skimage.feature import graycomatrix, graycoprops

from obia.features import (
    GLCM_FEATURES, TextureConfig, compute_index, extract_features,
)
from obia.handlers.geotif import Image, open_geotiff
from obia.segmentation import labels_to_segments
from obia.segmentation.segment import segment
from obia.segmentation.segment_statistics import (
    calculate_spectral_stats, calculate_textural_stats, create_objects,
)


def make_image(data, transform=Affine(2, 0, 100, 0, -3, 200), crs="EPSG:32605", valid_mask=None):
    data = np.asanyarray(data)
    if data.ndim == 2:
        data = data[:, :, None]
    return Image(data, crs, [transform.a, transform.b, transform.d, transform.e,
                           transform.c, transform.f], transform, None, valid_mask)


def whole_image(image):
    labels = np.ones(image.img_data.shape[:2], dtype=np.int32)
    return labels_to_segments(labels, image)


def texture_result(values, *, config=None, valid_mask=None):
    image = make_image(np.asarray(values, dtype=float))
    return extract_features(whole_image(image), image, spectral=(), percentiles=(), mask=valid_mask,
                            texture=config or TextureConfig()).iloc[0]


def test_distribution_values_and_invalid_sample_counts():
    data = np.array([[1, 2, 3], [4, np.nan, np.inf]], dtype=float)
    image = make_image(data)
    before = image.img_data.copy()
    objects = extract_features(whole_image(image), image, percentiles=(0, 10, 25, 75, 90, 100))
    row = objects.iloc[0]
    assert row.b0_mean == 2.5
    assert row.b0_median == 2.5
    assert row.b0_std == pytest.approx(np.sqrt(1.25))
    assert row.b0_variance == 1.25
    assert row.b0_iqr == 1.5
    assert row.b0_mad == 1
    assert row.b0_min == row.b0_p0 == 1
    assert row.b0_max == row.b0_p100 == 4
    assert row.b0_p10 == pytest.approx(1.3)
    assert row.b0_p90 == pytest.approx(3.7)
    assert row.b0_skewness == 0
    assert row.b0_kurtosis == pytest.approx(-1.36)
    assert row.b0_valid_count == 4
    assert row.b0_valid_fraction == pytest.approx(4 / 6)
    np.testing.assert_array_equal(image.img_data, before)


def test_valid_fraction_denominator_excludes_polygon_holes_but_includes_masked_pixels():
    image = make_image(np.ones((5, 5)))
    labels = np.ones((5, 5), dtype=np.int32)
    labels[2, 2] = 0
    included = np.ones((5, 5), dtype=bool)
    included[0, 0] = False
    row = extract_features(labels_to_segments(labels, image), image, mask=included).iloc[0]
    assert row.b0_valid_count == 23
    assert row.b0_valid_fraction == pytest.approx(23 / 24)


def test_masked_arrays_constant_bands_and_all_invalid_objects():
    masked = np.ma.array(np.ones((3, 3, 1)), mask=False)
    masked.mask[0, 0] = True
    image = make_image(masked)
    row = extract_features(whole_image(image), image).iloc[0]
    assert row.b0_valid_count == 8
    assert row.b0_std == row.b0_iqr == row.b0_mad == 0
    assert np.isnan(row.b0_skewness) and np.isnan(row.b0_kurtosis)
    objects = extract_features(whole_image(image), image, mask=np.zeros((3, 3), dtype=bool),
                               texture=TextureConfig(entropy_radii=(1,)))
    row = objects.iloc[0]
    assert row.b0_valid_count == row.b0_valid_fraction == 0
    for col in objects.columns:
        if col.startswith("b0_") and col not in {"b0_valid_count", "b0_valid_fraction"}:
            assert np.isnan(row[col])


@pytest.mark.parametrize("method", ["normalized_difference", "ndvi", "gndvi", "ndre", "ndwi",
                                    "mndwi", "ndmi", "nbr", "ndbi"])
def test_normalized_difference_indices_use_ordered_numeric_bands(method):
    values = np.array([[[200, 7, 50], [100, 8, 300]]], dtype=np.uint16)
    before = values.copy()
    actual = compute_index(values, method, (2, 0))
    np.testing.assert_allclose(actual, [[-0.6, 0.5]])
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("method,bands,expected", [
    ("ratio", (2, 0), 4), ("difference", (0, 2), -6),
    ("exg", (0, 1, 2), 2 / 16), ("vari", (0, 2, 1), 6 / 4),
    ("brightness", (0, 2), 5),
])
def test_other_spectral_indices(method, bands, expected):
    result = compute_index(np.array([[[2, 6, 8]]]), method, bands)
    assert result[0, 0] == pytest.approx(expected)


def test_invalid_index_operands_and_denominators_produce_nan():
    values = np.array([[[0, 0], [np.nan, 1], [np.inf, 1], [1, -1]]])
    assert np.isnan(compute_index(values, "ndvi", (0, 1))).all()
    assert np.isnan(compute_index(values, "ratio", (1, 0))[0, :3]).all()


def test_indices_are_summarized_after_pixel_calculation_and_use_joint_validity():
    image = make_image(np.array([[[1, 3], [2, 10], [np.nan, 7]]]))
    row = extract_features(whole_image(image), image, bands=(), indices={"ndvi": (1, 0)}).iloc[0]
    assert row.index_ndvi_b1_b0_mean == pytest.approx((0.5 + 2 / 3) / 2)
    assert row.index_ndvi_b1_b0_mean != pytest.approx((6.5 - 1.5) / (6.5 + 1.5))
    assert row.index_ndvi_b1_b0_valid_count == 2
    assert row.index_ndvi_b1_b0_valid_fraction == pytest.approx(2 / 3)


def test_source_mask_and_reordered_geotiff_bands_are_respected(tmp_path):
    path = tmp_path / "selected.tif"
    values = np.stack([np.full((4, 4), 10, dtype=np.uint16),
                       np.full((4, 4), 20, dtype=np.uint16),
                       np.full((4, 4), 70, dtype=np.uint16)])
    values[2, 0, 0] = 65535
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=3, dtype="uint16",
                       nodata=65535, crs="EPSG:32605", transform=Affine(2, 0, 100, 0, -3, 200)) as dst:
        dst.write(values)
    image = open_geotiff(path, bands=[3, 1])
    # Feature extraction is independent of the open file and uses loaded band order.
    image.rasterio_obj.close()
    row = extract_features(whole_image(image), image, indices={"ratio": (0, 1)},
                           texture=TextureConfig(bands=(0,))).iloc[0]
    assert row.b0_mean == 70 and row.b1_mean == 10
    assert row.b0_valid_count == 15 and row.b1_valid_count == 16
    assert row.index_ratio_b0_b1_mean == 7
    assert row.index_ratio_b0_b1_valid_count == 15
    assert row.b0_glcm_contrast_d1 == 0
    assert image.img_data[0, 0, 0] == 65535


def test_legacy_texture_axis_regression_and_explicit_channels_last():
    cube = np.stack([np.full((8, 8), value, dtype=float) for value in (10, 100, 200)])
    correct = calculate_textural_stats(cube, [0, 1, 2])
    for band in range(3):
        assert correct[f"b{band}_contrast"] == 0
        assert correct[f"b{band}_homogeneity"] == 1
    assert calculate_textural_stats(np.moveaxis(cube, 0, -1), [0, 1, 2], channel_axis=-1) == correct


def test_legacy_spectral_helper_preserves_band_first_inputs_and_custom_statistics():
    data = np.array([[[1, 2], [3, np.nan]], [[10, 10], [10, 10]]])
    original = data.copy()
    result = calculate_spectral_stats(data, [1, 0], features=("median", "valid_count", "valid_fraction"),
                                      percentiles=(25,), mask=np.array([[True, True], [False, True]]))
    assert result["b1_median"] == 10 and result["b0_median"] == 1.5
    assert result["b0_valid_count"] == 2 and result["b0_valid_fraction"] == pytest.approx(2 / 3)
    assert result["b0_p25"] == 1.25
    assert "b0_mean" not in result
    np.testing.assert_array_equal(data, original)


def test_masked_glcm_matches_hand_calculated_pairs_without_background_edges():
    # Two real pairs: 1->1 and 1->2. All pairs touching the third column are invalid.
    values = np.array([[1, 1, 99], [1, 2, 99]], dtype=float)
    valid = np.array([[True, True, False], [True, True, False]])
    config = TextureConfig(features=GLCM_FEATURES, angles=(0,), levels=3, value_range=(0, 3))
    row = texture_result(values, config=config, valid_mask=valid)
    assert row.b0_glcm_contrast_d1 == 0.5
    assert row.b0_glcm_dissimilarity_d1 == 0.5
    assert row.b0_glcm_homogeneity_d1 == 0.75
    assert row.b0_glcm_ASM_d1 == 0.375
    assert row.b0_glcm_energy_d1 == pytest.approx(np.sqrt(0.375))
    assert row.b0_glcm_entropy_d1 == 1.5
    assert row.b0_glcm_mean_d1 == 1.25
    assert row.b0_glcm_variance_d1 == 0.1875
    assert row.b0_glcm_maximum_probability_d1 == 0.5
    assert row.b0_glcm_cluster_shade_d1 == 0
    assert row.b0_glcm_cluster_prominence_d1 == 0.0625


@pytest.mark.parametrize("symmetric", [False, True])
def test_glcm_directions_and_distances_match_scikit_image(symmetric):
    values = np.array([[0, 1, 2, 3], [3, 2, 0, 1], [2, 0, 3, 1], [1, 3, 2, 0]], dtype=np.uint8)
    config = TextureConfig(features=GLCM_FEATURES[:6], distances=(1, 2),
                           angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, -np.pi / 4),
                           levels=4, value_range=(0, 4), symmetric=symmetric, angle_aggregation="none")
    row = texture_result(values, config=config)
    matrix = graycomatrix(values, config.distances, config.angles, levels=4,
                          symmetric=symmetric, normed=True)
    for feature in config.features:
        expected = graycoprops(matrix, feature)
        for d, distance in enumerate(config.distances):
            for a in range(len(config.angles)):
                assert row[f"b0_glcm_{feature}_d{distance}_a{a}"] == pytest.approx(expected[d, a])


def test_glcm_has_no_pairs_for_singletons_and_missing_directions_are_not_zeros():
    singleton = texture_result([[2]])
    assert np.isnan(singleton.b0_glcm_contrast_d1)
    row = texture_result([[0, 1]], config=TextureConfig(angles=(0, np.pi / 2), levels=2, value_range=(0, 2)))
    assert row.b0_glcm_contrast_d1 == 1  # average only the direction with samples


def test_local_entropy_ignores_invalid_pixels_and_reports_multiple_radii():
    values = np.array([[0, 1], [1, 0]], dtype=float)
    config = TextureConfig(features=(), entropy_radii=(1, 2), levels=2, value_range=(0, 2))
    row = texture_result(values, config=config)
    expected_r1 = -(1 / 3 * np.log2(1 / 3) + 2 / 3 * np.log2(2 / 3))
    assert row.b0_local_entropy_r1_mean == pytest.approx(expected_r1)
    assert row.b0_local_entropy_r2_mean == 1
    assert row.b0_local_entropy_r2_std == 0
    # Only one included pixel: padding/other-object pixels must contribute nothing.
    row = texture_result(values, config=config, valid_mask=np.array([[True, False], [False, False]]))
    assert row.b0_local_entropy_r1_mean == row.b0_local_entropy_r2_mean == 0


def test_quantization_is_shared_across_objects_and_uint16_is_not_wrapped():
    image = make_image(np.array([[1000, 2000, 5000, 6000]], dtype=np.uint16))
    segments = labels_to_segments(np.array([[1, 1, 2, 2]], dtype=np.int32), image)
    config = TextureConfig(angles=(0,), levels=6)
    result = extract_features(segments, image, spectral=(), percentiles=(), texture=config)
    np.testing.assert_array_equal(result.b0_glcm_contrast_d1, [1, 1])
    assert result.attrs["obia_features"]["texture_ranges"][0] == (1000, 6000)
    fixed = extract_features(segments, image, spectral=(), percentiles=(),
                             texture=TextureConfig(angles=(0,), levels=6, value_range=(0, 12000)))
    assert fixed.attrs["obia_features"]["texture_ranges"][0] == (0, 12000)


def test_polygon_holes_do_not_create_texture_edges():
    data = np.full((7, 7), 100, dtype=float)
    data[2:5, 2:5] = 0
    labels = np.ones((7, 7), dtype=np.int32)
    labels[2:5, 2:5] = 0
    image = make_image(data)
    result = extract_features(labels_to_segments(labels, image), image, spectral=(), percentiles=(),
                              texture=TextureConfig(entropy_radii=(1, 2)))
    row = result.iloc[0]
    assert row.b0_glcm_contrast_d1 == 0
    assert row.b0_glcm_entropy_d1 == 0
    assert row.b0_local_entropy_r2_mean == 0


def test_shape_features_on_rectangle_with_map_and_pixel_units():
    image = make_image(np.ones((2, 4)))
    objects = whole_image(image)
    row = extract_features(objects, image, spectral=(), percentiles=(), shape=True).iloc[0]
    assert row.shape_area == 48 and row.shape_perimeter == 28
    assert row.shape_major_axis_length == 8 and row.shape_minor_axis_length == 6
    assert row.shape_elongation == pytest.approx(8 / 6)
    assert row.shape_circularity == pytest.approx(4 * np.pi * 48 / 28 ** 2)
    assert row.shape_solidity == row.shape_rectangularity == 1
    assert row.shape_orientation == row.shape_hole_count == 0
    pixel = extract_features(objects, image, spectral=(), percentiles=(), shape=True, shape_units="pixel").iloc[0]
    assert pixel.shape_area == 8 and pixel.shape_perimeter == 12
    assert pixel.shape_major_axis_length == 4 and pixel.shape_minor_axis_length == 2


def test_shape_holes_and_concavity():
    image = make_image(np.ones((5, 5)), transform=Affine.identity())
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]])
    objects = gpd.GeoDataFrame({"segment_id": [4]}, geometry=[polygon], crs=image.crs)
    row = extract_features(objects, image, spectral=(), percentiles=(), shape=True).iloc[0]
    assert row.shape_area == 12
    assert row.shape_perimeter == 24
    assert row.shape_hole_count == 1
    assert row.shape_solidity == row.shape_rectangularity == 0.75


def test_rotated_and_sheared_raster_uses_full_affine_for_pixels_and_statistics():
    transform = Affine.translation(100, 200) * Affine.rotation(30) * Affine.shear(10) * Affine.scale(2, -3)
    data = np.arange(24).reshape(4, 6)
    image = make_image(data, transform=transform)
    objects = whole_image(image)
    row = extract_features(objects, image, shape=True, shape_units="pixel").iloc[0]
    assert row.b0_mean == pytest.approx(data.mean())
    assert row.b0_valid_count == 24
    assert row.shape_area == pytest.approx(24)
    assert row.shape_major_axis_length == pytest.approx(6)
    assert row.shape_minor_axis_length == pytest.approx(4)


def test_orientation_of_rotated_rectangle():
    image = make_image(np.ones((10, 10)), transform=Affine.identity())
    objects = gpd.GeoDataFrame(geometry=[rotate(box(0, 0, 4, 2), 30)], crs=image.crs)
    row = extract_features(objects, image, spectral=(), percentiles=(), shape=True).iloc[0]
    assert row.shape_orientation == pytest.approx(30)


def test_geographic_shape_requires_explicit_pixel_units():
    image = make_image(np.ones((2, 3)), crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected CRS"):
        extract_features(whole_image(image), image, shape=True)
    row = extract_features(whole_image(image), image, shape=True, shape_units="pixel").iloc[0]
    assert row.shape_area == pytest.approx(6)


def test_preserve_ids_index_attributes_geometry_and_recompute_columns():
    image = make_image(np.arange(8).reshape(2, 4))
    objects = whole_image(image)
    objects.index = [17]
    objects["segment_id"] = 99
    objects["class"] = 2
    objects["b0_mean"] = -1
    before = objects.copy()
    output = extract_features(objects, image, spectral=("mean",), percentiles=())
    assert output.index.tolist() == [17]
    assert output.iloc[0].segment_id == 99 and output.iloc[0]["class"] == 2
    assert output.iloc[0].b0_mean == 3.5
    assert output.geometry.equals(objects.geometry) and output.crs == objects.crs
    assert objects.equals(before)


def test_empty_tables_and_objects_outside_raster_have_stable_columns():
    image = make_image(np.ones((3, 3)))
    empty = whole_image(image).iloc[:0]
    output = extract_features(empty, image, shape=True, texture=TextureConfig())
    assert output.empty and output.crs == image.crs
    assert {"b0_mean", "shape_area", "b0_glcm_entropy_d1"} <= set(output)
    outside = gpd.GeoDataFrame({"segment_id": [8]}, geometry=[box(1000, 1000, 1002, 1002)], crs=image.crs)
    row = extract_features(outside, image, shape=True).iloc[0]
    assert row.b0_valid_count == 0 and np.isnan(row.b0_mean) and np.isnan(row.b0_valid_fraction)
    assert row.shape_area == 4


def test_legacy_flags_and_shape_only_requests():
    image = make_image(np.arange(16).reshape(4, 4))
    objects = whole_image(image)
    only_shape = create_objects(objects, image, calculate_spectral=False, calculate_textural=False,
                                shape_features=("area",))
    assert set(only_shape) == {"segment_id", "geometry", "shape_area"}
    only_mean = create_objects(objects, image, calculate_textural=False, calc_variance=False,
                               calc_min=False, calc_max=False, calc_skewness=False, calc_kurtosis=False)
    assert set(only_mean) == {"segment_id", "geometry", "b0_mean"}
    only_texture = create_objects(objects, image, calculate_spectral=False)
    assert "b0_mean" not in only_texture and "b0_contrast" in only_texture
    assert not any("glcm_" in col for col in only_texture)


def test_segment_integration_routes_features_without_passing_them_to_algorithm():
    data = np.stack([np.full((12, 12), 2), np.full((12, 12), 6)], axis=-1)
    image = make_image(data)
    result = segment(image, segmentation_bands=[0], statistics_bands=[1], n_segments=4,
                     feature_options={"spectral_features": ("mean", "median"), "percentiles": (25,),
                                      "indices": {"ndvi": (1, 0)}, "shape_features": True,
                                      "texture_config": TextureConfig(bands=(1,), entropy_radii=(1,))})
    assert "b0_mean" not in result.segments
    assert np.all(result.segments.b1_median == 6)
    assert np.all(result.segments.index_ndvi_b1_b0_mean == 0.5)
    assert result.segments.shape_area.sum() == pytest.approx(12 * 12 * 6)
    assert np.all(result.segments.b1_glcm_contrast_d1 == 0)
    assert np.all(result.segments.b1_local_entropy_r1_mean == 0)


@pytest.mark.parametrize("kwargs,error,match", [
    ({"bands": ["red"]}, TypeError, "integer indices"),
    ({"bands": [True]}, TypeError, "integer indices"),
    ({"bands": [0, 0]}, ValueError, "repeated"),
    ({"bands": [4]}, IndexError, "out of range"),
    ({"spectral": ["imaginary"]}, ValueError, "spectral features"),
    ({"spectral": "mean"}, TypeError, "sequence"),
    ({"percentiles": [-1]}, ValueError, "between 0 and 100"),
    ({"percentiles": [20, 20]}, ValueError, "distinct"),
    ({"indices": {"unknown": (0, 1)}}, ValueError, "Unknown spectral index"),
    ({"indices": {"ndvi": (0,)}}, ValueError, "exactly 2"),
    ({"indices": {"ndvi": (0, "nir")}}, TypeError, "integer indices"),
    ({"indices": {"ndvi": None}}, TypeError, "explicit numeric"),
    ({"shape": ["unknown"]}, ValueError, "shape features"),
    ({"mask": np.ones((2, 2))}, ValueError, "mask must"),
    ({"texture": {}}, TypeError, "TextureConfig"),
    ({"texture": TextureConfig(bands=(3,))}, IndexError, "out of range"),
])
def test_invalid_feature_requests(kwargs, error, match):
    image = make_image(np.ones((3, 3, 2)))
    with pytest.raises(error, match=match):
        extract_features(whole_image(image), image, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"distances": (0,)}, {"distances": (1.2,)}, {"levels": 1}, {"levels": 257},
    {"levels": True}, {"value_range": (10, 0)}, {"value_range": (0, np.nan)},
    {"angles": (np.nan,)}, {"angles": ()}, {"angle_aggregation": "sum"},
    {"entropy_radii": (-1,)}, {"features": ("unknown",)},
])
def test_invalid_texture_settings(kwargs):
    with pytest.raises(ValueError):
        TextureConfig(**kwargs)


def test_mismatched_crs_is_rejected():
    image = make_image(np.ones((2, 3)))
    objects = whole_image(image).set_crs("EPSG:4326", allow_override=True)
    with pytest.raises(ValueError, match="same CRS"):
        extract_features(objects, image)
