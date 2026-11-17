"""Checkerboard coverage, protected objects, and Quickshift pseudo-mask regressions."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from shapely.affinity import affine_transform
from shapely.geometry import box
from skimage import segmentation as skseg

from obia.utils import tiling


TRANSFORM = Affine(2, 0, 100, 0, -3, 200)
CRS = "EPSG:32605"


def write_raster(path, data, *, transform=TRANSFORM, crs=CRS, nodata=None):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None, :, :]
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=data.dtype, transform=transform,
                       crs=crs, nodata=nodata) as dst:
        dst.write(data)
    return path


def options(method):
    if method == "slic":
        return dict(n_segments=12, compactness=0.2, convert2lab=False)
    return dict(kernel_size=1, max_dist=3, ratio=0.5, convert2lab=False)


def assert_coverage(output_dir, expected, transform=TRANSFORM, crs=CRS):
    objects = gpd.read_file(output_dir / "segments.gpkg")
    assert objects.crs == crs
    assert objects.segment_id.is_unique
    assert list(objects.segment_id) == list(range(1, len(objects) + 1))
    counts = np.zeros(expected.shape, dtype="uint32")
    if len(objects):
        assert objects.geometry.is_valid.all()
        assert (objects.geometry.geom_type == "Polygon").all()
        counts = rasterize([(geometry, 1) for geometry in objects.geometry],
                           out_shape=expected.shape, transform=transform,
                           merge_alg=MergeAlg.add, dtype="uint32")
    # Addition, rather than overwriting labels, detects overlapping polygons too.
    np.testing.assert_array_equal(counts, expected.astype("uint32"))
    assert sum(geometry.area for geometry in objects.geometry) == pytest.approx(
        expected.sum() * abs(transform.determinant))
    return objects


@pytest.mark.parametrize("method", ["slic", "quickshift"])
@pytest.mark.parametrize("shape", [(37, 41), (7, 6), (1, 25), (25, 1), (1, 1)])
@pytest.mark.parametrize("masked", [False, True])
def test_every_included_pixel_is_covered_once(tmp_path, method, shape, masked):
    data = np.random.default_rng(4).normal(size=(3, *shape)).astype("float32")
    source = write_raster(tmp_path / "source.tif", data)
    valid = np.ones(shape, dtype=bool)
    mask_path = None
    if masked:
        valid[2:5, 3:9] = False
        valid[9:15, 10:18] = False  # Hole across a tile corner.
        valid[-1, -1] = False
        mask_path = write_raster(tmp_path / "mask.tif", valid.astype("uint8"))
    output = tmp_path / "output"
    assert tiling.create_tiled_segments(source, output, input_mask=mask_path, method=method,
                                        tile_size=12, buffer=4, **options(method)) is None
    assert_coverage(output, valid)
    with rasterio.open(source) as reopened:
        np.testing.assert_array_equal(reopened.read(), data)


@pytest.mark.parametrize("method", ["slic", "quickshift"])
@pytest.mark.parametrize("transform", [
    Affine.translation(100.123, 200.456) * Affine.rotation(17) * Affine.scale(0.3, -0.7),
    Affine(0.00001, 0, -155, 0, -0.00002, 20),
])
def test_rotated_fractional_and_geographic_grids(tmp_path, method, transform):
    crs = "EPSG:4326" if transform.c == -155 else CRS
    data = np.random.default_rng(1).random((29, 34)).astype("float32")
    source = write_raster(tmp_path / "source.tif", data, transform=transform, crs=crs)
    valid = np.ones(data.shape, dtype=bool)
    valid[7:19, 8:15] = False
    mask = write_raster(tmp_path / "mask.tif", valid.astype("uint8"), transform=transform, crs=crs)
    output = tmp_path / "output"
    tiling.create_tiled_segments(source, output, input_mask=mask, method=method,
                                 tile_size=12, buffer=4, **options(method))
    assert_coverage(output, valid, transform=transform, crs=crs)


@pytest.mark.parametrize("ratio", [0, 0.001, 0.5])
@pytest.mark.parametrize("sigma", [0, 1])
def test_quickshift_mixed_fill_and_valid_pixels_never_leave_gaps(tmp_path, ratio, sigma):
    # A spatial-only run (ratio=0) deliberately cannot distinguish the fill.
    data = np.zeros((28, 31), dtype="float32")
    valid = np.ones(data.shape, dtype=bool)
    valid[5:24, 13:15] = False
    valid[11:14, 5:24] = False
    source = write_raster(tmp_path / "source.tif", data)
    mask = write_raster(tmp_path / "mask.tif", valid.astype("uint8"))
    output = tmp_path / "output"
    tiling.create_tiled_segments(source, output, input_mask=mask, method="quickshift",
                                 tile_size=12, buffer=4, kernel_size=2, max_dist=6,
                                 ratio=ratio, sigma=sigma, convert2lab=False)
    assert_coverage(output, valid)


def test_whole_replacements_cross_tile_bounds_and_crossing_objects_are_unchanged(tmp_path, monkeypatch):
    calls = []

    def mixed_quickshift(data, **kwargs):
        calls.append(data.copy())
        if len(calls) == 1:
            rows, cols = np.indices(data.shape[:2])
            return (rows // 3) * 4 + cols // 3
        # Simulate a single label containing protected, real, and fill pixels.
        return np.zeros(data.shape[:2], dtype=int)

    monkeypatch.setattr(skseg, "quickshift", mixed_quickshift)
    source = write_raster(tmp_path / "source.tif", np.zeros((12, 24), dtype="float32"))
    output = tmp_path / "output"
    tiling.create_tiled_segments(source, output, method="quickshift", tile_size=12, buffer=4)
    objects = assert_coverage(output, np.ones((12, 24), dtype=bool))
    coefficients = [TRANSFORM.a, TRANSFORM.b, TRANSFORM.d, TRANSFORM.e, TRANSFORM.c, TRANSFORM.f]
    expected_crossing = affine_transform(box(9, 0, 24, 12), coefficients)
    protected = affine_transform(box(6, 0, 9, 3), coefficients)
    assert any(geometry.equals(expected_crossing) for geometry in objects.geometry)
    assert any(geometry.equals(protected) for geometry in objects.geometry)
    assert len(calls) == 2
    assert (calls[1][:, 0] < 0).all()  # Protected strip gets a distinct fill.
    assert (calls[1][:, 1:] == 0).all()  # Real zeros remain valid.


@pytest.mark.parametrize("method", ["slic", "quickshift"])
def test_nodata_nonfinite_and_unused_bands(tmp_path, method):
    data = np.ones((3, 27, 29), dtype="float32")
    data[0, 3:7, 5:11] = -9999
    data[1, 12:16, 9:19] = np.nan
    data[1, 24, 24] = np.inf
    data[2] = np.nan  # An unselected band must not exclude the whole scene.
    valid = np.all(np.isfinite(data[:2]) & (data[:2] != -9999), axis=0)
    source = write_raster(tmp_path / "source.tif", data, nodata=-9999)
    output = tmp_path / "output"
    tiling.create_tiled_segments(source, output, method=method, tile_size=12, buffer=4,
                                 segmentation_bands=[0, 1], **options(method))
    assert_coverage(output, valid)


@pytest.mark.parametrize("method", ["slic", "quickshift"])
def test_empty_tiles_and_isolated_pixels(tmp_path, method):
    data = np.zeros((31, 29), dtype="float32")
    valid = np.zeros(data.shape, dtype=bool)
    valid[0, 0] = valid[12, 12] = valid[-1, -1] = True
    source = write_raster(tmp_path / "source.tif", data)
    mask = write_raster(tmp_path / "mask.tif", valid.astype("uint8"))
    output = tmp_path / "output"
    tiling.create_tiled_segments(source, output, input_mask=mask, method=method,
                                 tile_size=12, buffer=4, **options(method))
    objects = assert_coverage(output, valid)
    assert len(objects) == 3
    empty = write_raster(tmp_path / "empty.tif", np.zeros_like(valid, dtype="uint8"))
    tiling.create_tiled_segments(source, output, input_mask=empty, method=method,
                                 tile_size=12, buffer=4, **options(method))
    assert_coverage(output, np.zeros_like(valid))


def test_slic_without_explicit_segment_count_or_mask(tmp_path):
    source = write_raster(tmp_path / "source.tif", np.ones((27, 25), dtype="uint16"))
    output = tmp_path / "output"
    tiling.create_tiled_segments(source, output, tile_size=12, buffer=4, crown_radius=3)
    assert_coverage(output, np.ones((27, 25), dtype=bool))


@pytest.mark.parametrize("difference", ["shape", "transform", "crs"])
def test_misaligned_masks_are_rejected(tmp_path, difference):
    source = write_raster(tmp_path / "source.tif", np.ones((24, 24), dtype="uint8"))
    shape = (23, 24) if difference == "shape" else (24, 24)
    transform = TRANSFORM * Affine.translation(1, 0) if difference == "transform" else TRANSFORM
    crs = "EPSG:4326" if difference == "crs" else CRS
    mask = write_raster(tmp_path / "mask.tif", np.ones(shape, dtype="uint8"),
                        transform=transform, crs=crs)
    with pytest.raises(ValueError, match="dimensions, CRS, and transform"):
        tiling.create_tiled_segments(source, tmp_path / "output", input_mask=mask)


@pytest.mark.parametrize("failure", ["exception", "missing_pixels", "excluded_pixels"])
def test_failed_replacement_preserves_objects_and_existing_output(tmp_path, monkeypatch, failure):
    original = tiling.create_labels
    calls = 0
    stores = []
    original_replace = tiling._SegmentStore.replace

    def remember_store(self, *args):
        original_replace(self, *args)
        stores.append((self, dict(self.geometries)))

    def fail_replacement(image, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            if failure == "exception":
                raise ValueError("simulated Quickshift failure")
            if failure == "missing_pixels":
                return np.zeros(image.img_data.shape[:2], dtype="int32")
            return np.ones(image.img_data.shape[:2], dtype="int32")
        return original(image, **kwargs)

    monkeypatch.setattr(tiling._SegmentStore, "replace", remember_store)
    monkeypatch.setattr(tiling, "create_labels", fail_replacement)
    source = write_raster(tmp_path / "source.tif", np.ones((12, 24), dtype="float32"))
    mask = np.ones((12, 24), dtype="uint8")
    mask[4:6, 18:20] = 0
    mask_path = write_raster(tmp_path / "mask.tif", mask)
    output = tmp_path / "output"
    output.mkdir()
    old_output = output / "segments.gpkg"
    old_output.write_bytes(b"previous result")
    with pytest.raises((RuntimeError, ValueError), match="simulated|included pixels"):
        tiling.create_tiled_segments(source, output, input_mask=mask_path, method="quickshift",
                                     tile_size=12, buffer=4, **options("quickshift"))
    assert len(stores) == 1
    store, saved = stores[0]
    assert store.geometries == saved
    assert old_output.read_bytes() == b"previous result"


@pytest.mark.parametrize("kwargs,match", [
    ({"method": "watershed"}, "supports"),
    ({"tile_size": 0}, "tile_size"),
    ({"tile_size": 5.5}, "tile_size"),
    ({"tile_size": True}, "tile_size"),
    ({"buffer": -1}, "buffer"),
    ({"buffer": 200}, "buffer"),
    ({"buffer": 2.5}, "buffer"),
    ({"n_segments": 0}, "n_segments"),
    ({"n_segments": 2.5}, "n_segments"),
    ({"crown_radius": 0}, "crown_radius"),
    ({"method": "quickshift", "n_segments": 5}, "SLIC option"),
    ({"mask": np.ones((2, 2))}, "input_mask"),
])
def test_invalid_tiling_options_fail_clearly(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        tiling.create_tiled_segments(tmp_path / "unused.tif", tmp_path / "output", **kwargs)


def test_source_is_read_in_selected_band_windows(tmp_path, monkeypatch):
    source = write_raster(tmp_path / "source.tif", np.zeros((4, 37, 41), dtype="float32"))
    original_read = tiling._read_tile
    windows = []

    def record_read(source, mask_source, window, indexes):
        assert indexes == [4, 2]
        assert window.width <= 20 and window.height <= 20
        windows.append(window)
        image, valid = original_read(source, mask_source, window, indexes)
        assert image.img_data.shape[2] == 2
        return image, valid

    monkeypatch.setattr(tiling, "_read_tile", record_read)
    output = tmp_path / "output"
    tiling.create_tiled_segments(source, output, method="quickshift", segmentation_bands=[3, 1],
                                 tile_size=12, buffer=4, **options("quickshift"))
    assert len(windows) == 16
    assert_coverage(output, np.ones((37, 41), dtype=bool))
