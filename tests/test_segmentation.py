"""Exercise every supported algorithm and the raster-to-object contract."""

import numpy as np
import pytest
import rasterio
from affine import Affine
from PIL import Image as PILImage
from rasterio.features import rasterize
from skimage import segmentation as skseg
from skimage.draw import polygon2mask

from obia.handlers.geotif import Image, open_geotiff
from obia.segmentation import (
    SEGMENTATION_METHODS, create_labels, create_segments, labels_to_segments,
)
from obia.segmentation.segment import Segments, segment
from obia.segmentation.segment_boundaries import normalize_band


def make_image(data, crs="EPSG:32605", transform=None):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[:, :, None]
    transform = transform or Affine(2, 0, 100, 0, -3, 200)
    return Image(data, crs, [transform.a, transform.b, transform.d, transform.e,
                           transform.c, transform.f], transform, None)


@pytest.fixture
def image():
    rows, cols = np.mgrid[:24, :24]
    values = np.where((rows - 12) ** 2 + (cols - 12) ** 2 < 49, 200, 20)
    return make_image(values.astype(np.uint16))


def algorithm_params(method):
    markers = np.zeros((24, 24), dtype=np.int32)
    markers[3, 3] = 1
    markers[12, 12] = 2
    initial = np.zeros((24, 24), dtype=np.int8)
    initial[7:18, 7:18] = 1
    angle = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    snake = np.column_stack((12 + 8 * np.sin(angle), 12 + 8 * np.cos(angle)))
    return {
        "slic": dict(n_segments=8, compactness=0.2),
        "quickshift": dict(kernel_size=1, max_dist=3, ratio=0.5),
        "felzenszwalb": dict(scale=20, sigma=0.2, min_size=5),
        "watershed": dict(markers=markers, watershed_line=True),
        "random_walker": dict(labels=markers, beta=10, mode="bf"),
        "chan_vese": dict(max_num_iter=5, init_level_set="disk"),
        "morphological_chan_vese": dict(num_iter=5, init_level_set=initial),
        "morphological_geodesic_active_contour": dict(
            num_iter=3, init_level_set=initial, balloon=1, gradient_kwargs={"sigma": 1}),
        "active_contour": dict(snake=snake, max_num_iter=3),
        "flood": dict(seed_point=(12, 12), tolerance=0.1),
    }[method]


@pytest.mark.parametrize("method", SEGMENTATION_METHODS)
def test_each_algorithm_produces_georeferenced_objects_without_mutation(image, method):
    before = image.img_data.copy()
    labels = create_labels(image, method=method, **algorithm_params(method))
    objects = labels_to_segments(labels, image)
    assert labels.shape == (24, 24)
    assert labels.dtype == np.int32
    assert labels.max() > 0
    assert objects.crs == image.crs
    assert objects.geometry.is_valid.all()
    assert objects.segment_id.is_unique
    assert set(objects.segment_id) == set(np.unique(labels)) - {0}
    restored = rasterize(zip(objects.geometry, objects.segment_id), out_shape=labels.shape,
                         transform=image.transform, dtype="int32")
    np.testing.assert_array_equal(restored, labels)
    pixel_area = abs(image.transform.determinant)
    for row in objects.itertuples():
        assert row.geometry.area == pytest.approx((labels == row.segment_id).sum() * pixel_area)
    np.testing.assert_array_equal(image.img_data, before)


@pytest.mark.parametrize("method", SEGMENTATION_METHODS)
def test_masks_exclude_pixels_and_empty_masks_produce_no_objects(image, method):
    mask = np.ones((24, 24), dtype=bool)
    mask[:2] = False
    original_mask = mask.copy()
    labels = create_labels(image, method=method, mask=mask, **algorithm_params(method))
    np.testing.assert_array_equal(mask, original_mask)
    assert np.all(labels[~mask] == 0)
    assert labels.max() > 0
    empty = create_segments(image, method=method, mask=np.zeros_like(mask),
                            **algorithm_params(method))
    assert empty.empty
    assert empty.crs == image.crs
    assert list(empty.columns) == ["segment_id", "geometry"]


@pytest.mark.parametrize("method", SEGMENTATION_METHODS)
def test_public_segment_calculates_features_from_original_raster(tmp_path, image, method):
    path = tmp_path / "raster.tif"
    with rasterio.open(path, "w", driver="GTiff", width=24, height=24, count=1,
                       dtype="uint16", transform=image.transform, crs=image.crs) as dst:
        dst.write(image.img_data[:, :, 0], 1)
    source = open_geotiff(path)
    try:
        result = segment(source, method=method, **algorithm_params(method),
                         calc_skewness=False, calc_kurtosis=False, calc_contrast=False,
                         calc_dissimilarity=False, calc_homogeneity=False, calc_ASM=False,
                         calc_energy=False, calc_correlation=False)
        assert result.method == method
        assert result.labels is result._segments
        assert result.labels.shape == (24, 24)
        for row in result.segments.itertuples():
            expected = image.img_data[:, :, 0][result.labels == row.segment_id].mean()
            assert row.b0_mean == pytest.approx(expected)
        overlay = result.to_segmented_image(PILImage.new("RGB", (24, 24)))
        assert overlay.size == (24, 24)
    finally:
        source.rasterio_obj.close()


@pytest.mark.parametrize("method", ["slic", "quickshift", "felzenszwalb", "random_walker"])
@pytest.mark.parametrize("bands", [1, 3, 4])
def test_channel_methods_support_single_rgb_and_multispectral_bands(image, method, bands):
    image.img_data = np.repeat(image.img_data, bands, axis=2)
    labels = create_labels(image, method=method, **algorithm_params(method))
    assert labels.shape == (24, 24)
    assert np.all(labels > 0)


@pytest.mark.parametrize("method", ["slic", "quickshift", "felzenszwalb"])
def test_constant_integer_bands_are_finite_and_not_modified(method):
    image = make_image(np.full((12, 12, 1), 1500, dtype=np.uint16))
    with np.errstate(all="raise"):
        labels = create_labels(image, method=method)
    assert np.all(labels > 0)
    assert np.all(image.img_data == 1500)
    np.testing.assert_array_equal(normalize_band(image.img_data), 0)


def test_slic_zero_is_a_valid_segment_label(image):
    labels = create_labels(image, method="slic", start_label=0, n_segments=4)
    assert np.all(labels > 0)
    assert labels_to_segments(labels, image).geometry.area.sum() == 24 * 24 * 6


def test_flood_respects_native_units_when_normalization_is_disabled():
    data = np.array([[10, 11, 12], [10, 11, 13], [30, 30, 30]], dtype=np.uint16)
    image = make_image(data)
    labels = create_labels(image, method="flood", normalize=False, seed_point=(0, 0), tolerance=1)
    expected = skseg.flood(data, (0, 0), tolerance=1)
    np.testing.assert_array_equal(labels > 0, expected)


def test_binary_methods_do_not_polygonize_background(image):
    labels = create_labels(image, method="flood", seed_point=(12, 12))
    assert labels[0, 0] == 0
    assert labels[12, 12] > 0
    assert labels_to_segments(labels, image).geometry.area.sum() < 24 * 24 * 6


def test_random_walker_preserves_seeds_and_excludes_inactive_pixels(image):
    params = algorithm_params("random_walker")
    params["labels"][:2] = -1
    before = params["labels"].copy()
    labels = create_labels(image, method="random_walker", copy=False, **params)
    assert np.all(labels[:2] == 0)
    assert labels[3, 3] > 0 and labels[12, 12] > 0
    assert labels[3, 3] != labels[12, 12]
    np.testing.assert_array_equal(params["labels"], before)


def test_random_walker_masks_unsigned_markers_without_overflow(image):
    params = algorithm_params("random_walker")
    params["labels"] = params["labels"].astype(np.uint8)
    mask = np.ones((24, 24), dtype=bool)
    mask[:2] = False
    labels = create_labels(image, method="random_walker", mask=mask, **params)
    assert np.all(labels[:2] == 0)
    assert labels[3, 3] != labels[12, 12]
    assert params["labels"].dtype == np.uint8


def test_random_walker_ignores_nonfinite_inactive_pixels(image):
    image.img_data = image.img_data.astype(float)
    image.img_data[0] = np.nan
    params = algorithm_params("random_walker")
    params["labels"][0] = -1
    labels = create_labels(image, method="random_walker", **params)
    assert np.all(labels[0] == 0)
    assert np.all(labels[1:] > 0)


def test_precomputed_morphgac_gradient_matches_native_algorithm(image):
    gradient = skseg.inverse_gaussian_gradient(normalize_band(image.img_data[:, :, 0]), sigma=1)
    params = algorithm_params("morphological_geodesic_active_contour")
    params.pop("gradient_kwargs")
    # The upstream curvature operator cycles globally between calls. Disable
    # smoothing here to compare the precomputed gradient path deterministically.
    params["smoothing"] = 0
    expected = skseg.morphological_geodesic_active_contour(gradient, **params)
    actual = create_labels(image, method="morphological_geodesic_active_contour",
                           gimage=gradient, **params)
    np.testing.assert_array_equal(actual > 0, expected.astype(bool))


def test_active_contour_polygon_matches_native_snake(image):
    params = algorithm_params("active_contour")
    expected_snake = skseg.active_contour(normalize_band(image.img_data[:, :, 0]), **params)
    actual = create_labels(image, method="active_contour", **params)
    np.testing.assert_array_equal(actual > 0, polygon2mask((24, 24), expected_snake))


@pytest.mark.parametrize("bands", [3, 4])
def test_active_contour_accepts_multichannel_images(image, bands):
    image.img_data = np.repeat(image.img_data, bands, axis=2)
    labels = create_labels(image, method="active_contour", **algorithm_params("active_contour"))
    assert labels.shape == (24, 24)
    assert labels.max() > 0
    assert labels[0, 0] == 0


def test_disconnected_regions_get_distinct_ids_and_non_epsg_crs_is_preserved():
    image = make_image(np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]]),
                       crs="+proj=aeqd +lat_0=19.7 +lon_0=-155.1 +datum=WGS84 +units=m",
                       transform=Affine.translation(100, 200) * Affine.rotation(15) * Affine.scale(2, -3))
    labels = create_labels(image, method="watershed", markers=np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0]]),
                           mask=image.img_data[:, :, 0].astype(bool))
    objects = labels_to_segments(labels, image)
    assert len(objects) == 2
    assert objects.segment_id.is_unique
    assert objects.crs.to_epsg() is None
    assert objects.geometry.area.sum() == pytest.approx(36)


def test_masked_nonfinite_values_do_not_contaminate_valid_data(image):
    image.img_data = image.img_data.astype(float)
    image.img_data[0] = np.nan
    mask = np.ones((24, 24), dtype=bool)
    mask[0] = False
    labels = create_labels(image, mask=mask, n_segments=4)
    assert np.all(labels[0] == 0)
    assert np.all(labels[1:] > 0)
    with pytest.raises(ValueError, match="finite"):
        create_labels(image)


@pytest.mark.parametrize("method", ["watershed", "chan_vese", "morphological_chan_vese",
                                    "morphological_geodesic_active_contour", "flood"])
def test_scalar_algorithms_require_explicit_band_selection(image, method):
    image.img_data = np.repeat(image.img_data, 3, axis=2)
    with pytest.raises(ValueError, match="requires one band"):
        create_labels(image, method=method, **algorithm_params(method))
    assert create_labels(image, segmentation_bands=[1], method=method,
                         **algorithm_params(method)).shape == (24, 24)


@pytest.mark.parametrize("method,kwargs,match", [
    ("unknown", {}, "Unknown segmentation method"),
    ("random_walker", {}, "requires labels"),
    ("random_walker", {"labels": np.zeros((24, 24), dtype=int)}, "positive seed"),
    ("watershed", {"markers": np.zeros((3, 3), dtype=int)}, "markers must"),
    ("active_contour", {}, "requires snake"),
    ("active_contour", {"snake": [[1, 1], [2, 3], [3, 2]], "boundary_condition": "free"}, "periodic"),
    ("flood", {}, "requires seed_point"),
    ("flood", {"seed_point": (-1, 2)}, "inside the image"),
    ("flood", {"seed_point": 2}, "two integer"),
    ("morphological_chan_vese", {}, "requires num_iter"),
    ("morphological_geodesic_active_contour", {}, "requires num_iter"),
    ("slic", {"channel_axis": 0}, "channels last"),
    ("slic", {"mask": np.ones((2, 2))}, "mask must"),
    ("quickshift", {"return_tree": True}, "diagnostic"),
    ("chan_vese", {"extended_output": True}, "diagnostic"),
    ("random_walker", {"return_full_prob": True}, "diagnostic"),
])
def test_invalid_inputs_fail_clearly(image, method, kwargs, match):
    with pytest.raises(ValueError, match=match):
        create_labels(image, method=method, **kwargs)


@pytest.mark.parametrize("bands,error", [([], ValueError), ([-1], IndexError), ([1], IndexError),
                                        ([0.5], TypeError), ([True], TypeError)])
def test_invalid_band_indexes_do_not_modify_image(image, bands, error):
    before = image.img_data.copy()
    with pytest.raises(error):
        create_labels(image, segmentation_bands=bands)
    np.testing.assert_array_equal(image.img_data, before)


def test_result_parameters_are_not_shared():
    first = Segments(np.ones((2, 2)), None, "slic", n_segments=10)
    second = Segments(np.ones((2, 2)), None, "watershed", markers=2)
    assert first.params == {"n_segments": 10}
    assert second.params == {"markers": 2}
