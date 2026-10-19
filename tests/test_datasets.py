"""Check that bundled examples retain their source pixels and spatial grid."""

import hashlib

import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from obia.datasets import chesapeake_sample
from obia.handlers.geotif import Image, open_geotiff


@pytest.mark.parametrize("split", ["train", "test"])
def test_chesapeake_pair_matches_manifest_and_loads_as_image(split):
    sample = chesapeake_sample(split)
    metadata = sample["metadata"]
    with rasterio.open(sample["image"]) as image, rasterio.open(sample["reference"]) as reference:
        assert image.shape == reference.shape == tuple(metadata["shape"])
        assert image.crs == reference.crs == rasterio.crs.CRS.from_string(metadata["crs"])
        assert image.transform == reference.transform
        assert list(image.transform)[:6] == metadata["transform"]
        assert image.res == reference.res == (1, 1)
        assert image.count == 4 and reference.count == 1
        assert image.dtypes == ("uint8",) * 4 and reference.dtypes == ("uint8",)
        assert reference.nodata == 15
        codes, counts = np.unique(reference.read(1), return_counts=True)
        assert set(codes) == set(sample["class_names"])
        assert {str(code): int(count) for code, count in zip(codes, counts)} == metadata["class_pixel_counts"]
        for kind, src in (("image", image), ("reference", reference)):
            entry = metadata["files"][kind]
            assert hashlib.sha256(sample[kind].read_bytes()).hexdigest() == entry["sha256"]
            assert hashlib.sha256(src.read().tobytes()).hexdigest() == entry["pixel_sha256"]
            assert src.read_masks().all()
        loaded = open_geotiff(sample["image"])
        try:
            assert isinstance(loaded, Image)
            np.testing.assert_array_equal(loaded.img_data, image.read().transpose(1, 2, 0))
            assert loaded.valid_mask.all()
        finally:
            loaded.rasterio_obj.close()


def test_chesapeake_train_and_test_are_separate_source_tiles_and_areas():
    train, test = chesapeake_sample("train"), chesapeake_sample("test")
    assert train["metadata"]["upstream_split"] == "md-train"
    assert test["metadata"]["upstream_split"] == "md-test"
    assert train["metadata"]["tile_id"] != test["metadata"]["tile_id"]
    with rasterio.open(train["image"]) as a, rasterio.open(test["image"]) as b:
        assert a.crs == b.crs
        assert box(*a.bounds).distance(box(*b.bounds)) > 1000


def test_chesapeake_rejects_unknown_split():
    with pytest.raises(ValueError, match="split must be"):
        chesapeake_sample("validation")
