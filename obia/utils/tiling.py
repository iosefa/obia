"""Windowed checkerboard segmentation with protected objects in tile overlaps."""

from collections import defaultdict
from contextlib import ExitStack
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from affine import Affine
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.affinity import affine_transform

from obia.handlers.geotif import Image
from obia.segmentation.segment_boundaries import create_labels, labels_to_segments


def get_raster_bbox(dataset):
    """Return (min_x, min_y, max_x, max_y) for a GDAL raster dataset."""
    transform = Affine.from_gdal(*dataset.GetGeoTransform())
    corners = [transform * (col, row)
               for col in (0, dataset.RasterXSize)
               for row in (0, dataset.RasterYSize)]
    xs, ys = zip(*corners)
    return min(xs), min(ys), max(xs), max(ys)


class _SegmentStore:
    """Index accepted objects on the integer pixel grid, avoiding CRS rounding.

    A tile-sized spatial hash lets each window inspect nearby objects without
    rebuilding an index or copying the complete polygon table after every tile.
    Pixel-space geometry also supports rotated and non-square raster pixels.
    """

    def __init__(self, tile_size):
        self.tile_size = tile_size
        self.geometries = {}
        self.cells = defaultdict(set)
        self.next_id = 1

    def _cells(self, bounds):
        left, top, right, bottom = bounds
        for row in range(math.floor(top) // self.tile_size,
                         (math.ceil(bottom) - 1) // self.tile_size + 1):
            for col in range(math.floor(left) // self.tile_size,
                             (math.ceil(right) - 1) // self.tile_size + 1):
                yield row, col

    def partition(self, bounds):
        """Separate replaceable whole objects from protected crossing objects."""
        candidates = set()
        for cell in self._cells(bounds):
            candidates.update(self.cells.get(cell, ()))
        left, top, right, bottom = bounds
        replaceable, protected = [], []
        for object_id in sorted(candidates):
            geometry = self.geometries[object_id]
            x0, y0, x1, y1 = geometry.bounds
            if x1 <= left or y1 <= top or x0 >= right or y0 >= bottom:
                continue
            if left <= x0 and top <= y0 and x1 <= right and y1 <= bottom:
                replaceable.append(object_id)
            else:
                protected.append(geometry)
        return replaceable, protected

    def replace(self, object_ids, geometries):
        # Call only after segmentation and coverage verification have succeeded.
        for object_id in object_ids:
            geometry = self.geometries.pop(object_id)
            for cell in self._cells(geometry.bounds):
                self.cells[cell].remove(object_id)
                if not self.cells[cell]:
                    del self.cells[cell]
        for geometry in geometries:
            object_id = self.next_id
            self.next_id += 1
            self.geometries[object_id] = geometry
            for cell in self._cells(geometry.bounds):
                self.cells[cell].add(object_id)


def _band_indexes(count, bands):
    bands = list(range(count)) if bands is None else list(bands)
    if not bands:
        raise ValueError("segmentation_bands must contain at least one band index.")
    for band in bands:
        if isinstance(band, (bool, np.bool_)) or not isinstance(band, (int, np.integer)):
            raise TypeError("segmentation_bands must contain integer band indexes.")
        if band < 0 or band >= count:
            raise IndexError(f"Band index {band} out of range; expected 0 to {count - 1}.")
    return [int(band) + 1 for band in bands]


def _read_tile(source, mask_source, window, indexes):
    pixels = source.read(indexes, window=window, masked=True)
    data = np.moveaxis(pixels.data, 0, -1)
    valid = np.all(~np.ma.getmaskarray(pixels) & np.isfinite(pixels.data), axis=0)
    if mask_source is not None:
        mask = mask_source.read(1, window=window, masked=True)
        valid &= ~np.ma.getmaskarray(mask) & np.isfinite(mask.data) & (mask.data != 0)
    # Keep polygons on an exact pixel grid until the final output transformation.
    transform = Affine.translation(window.col_off, window.row_off)
    image = Image(data, None, [transform.a, transform.b, transform.d, transform.e,
                              transform.c, transform.f], transform, None)
    return image, valid


def create_tiled_segments(input_raster, output_dir, input_mask=None,
                          method="slic", tile_size=200, buffer=30, crown_radius=5,
                          **kwargs):
    """Write ``segments.gpkg`` using two sequential checkerboard passes.

    Black tiles are segmented first. Buffered white tiles then replace existing
    objects wholly contained in their windows, allowing new objects to cross the
    original tile boundaries. Objects crossing a buffered window's outer edge
    stay intact and are masked out of the new segmentation. Existing objects are
    removed only after their replacement covers every included pixel.

    Output polygons have unique positive ``segment_id`` values and the source
    CRS. Disconnected pieces receive separate IDs. The function writes the file
    and returns None.

    Args:
        input_raster (str | Path): Raster path; only the current window and
            selected bands are read. Source NoData, raster masks, and nonfinite
            pixels are excluded.
        output_dir (str | Path): Directory for the merged ``segments.gpkg``.
            An existing file is replaced only after all tiles and the new file
            are complete.
        input_mask (str | Path | None): Optional raster aligned with the input's
            dimensions, CRS, and transform. Nonzero, finite, non-NoData pixels
            are included.
        method (str): ``"slic"`` (native mask) or ``"quickshift"`` (finite
            pseudo-mask followed by clipping every label to the saved boolean mask).
        tile_size (int): Positive integer tile width/height in pixels.
        buffer (int): Integer overlap width in pixels, from 0 to ``tile_size - 1``.
            Choose enough overlap to contain the objects you want to replace.
        crown_radius (float): SLIC radius in map units, used to estimate
            ``n_segments`` from included area when that parameter is absent.
            Ignored by Quickshift.
        **kwargs (Any): Arguments for :func:`obia.segmentation.create_labels`,
            including zero-based ``segmentation_bands``, ``normalize``, and algorithm options.
            Explicit SLIC ``n_segments`` applies to each window, capped at its
            included pixel count. For Quickshift, use ``kernel_size``, ``max_dist``,
            and ``ratio`` instead.

    Notes:
        This is sequential, not parallel. Raster working memory is bounded by a
        buffered tile; accepted object geometries accumulate in memory. Quickshift's
        fill can affect nearby boundaries, even though clipping preserves all
        included pixels. Per-window normalization and limited context can produce
        different boundaries from segmenting the whole image at once.
    """
    if method not in {"slic", "quickshift"}:
        raise ValueError("Tiled segmentation supports 'slic' and 'quickshift'.")
    if (isinstance(tile_size, (bool, np.bool_))
            or not isinstance(tile_size, (int, np.integer)) or tile_size <= 0):
        raise ValueError("tile_size must be a positive integer.")
    if (isinstance(buffer, (bool, np.bool_)) or not isinstance(buffer, (int, np.integer))
            or not 0 <= buffer < tile_size):
        raise ValueError("buffer must be an integer from 0 to tile_size - 1.")
    options = dict(kwargs)
    if "mask" in options:
        raise ValueError("Use input_mask with a raster path for tiled segmentation.")
    bands = options.pop("segmentation_bands", None)
    n_segments = options.pop("n_segments", None)
    if method == "quickshift" and n_segments is not None:
        raise ValueError("n_segments is a SLIC option; Quickshift uses kernel_size and max_dist.")
    if method == "slic":
        if n_segments is not None:
            if (isinstance(n_segments, (bool, np.bool_))
                    or not isinstance(n_segments, (int, np.integer)) or n_segments <= 0):
                raise ValueError("n_segments must be a positive integer.")
        elif not np.isscalar(crown_radius) or not np.isfinite(crown_radius) or crown_radius <= 0:
            raise ValueError("crown_radius must be a finite positive number.")

    store = _SegmentStore(tile_size)
    with ExitStack() as stack:
        # Rasterio is already a core dependency; no separate osgeo bindings needed.
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024))
        source = stack.enter_context(rasterio.open(input_raster))
        indexes = _band_indexes(source.count, bands)
        mask_source = None
        if input_mask is not None:
            mask_source = stack.enter_context(rasterio.open(input_mask))
            if (mask_source.shape != source.shape or mask_source.crs != source.crs
                    or mask_source.transform != source.transform):
                raise ValueError("input_mask must match the raster's dimensions, CRS, and transform.")
        pixel_area = abs(source.transform.determinant)
        if not np.isfinite(pixel_area) or pixel_area <= 0:
            raise ValueError("The raster must have a finite, invertible spatial transform.")

        for parity in (0, 1):
            for row in range(0, source.height, tile_size):
                for col in range(0, source.width, tile_size):
                    if (row // tile_size + col // tile_size) % 2 != parity:
                        continue
                    overlap = buffer if parity else 0
                    left, top = max(0, col - overlap), max(0, row - overlap)
                    right = min(source.width, col + tile_size + overlap)
                    bottom = min(source.height, row + tile_size + overlap)
                    window = Window(left, top, right - left, bottom - top)
                    image, valid = _read_tile(source, mask_source, window, indexes)
                    replaceable, protected = store.partition((left, top, right, bottom))
                    if protected:
                        protected_mask = rasterize(
                            [(geometry, 1) for geometry in protected],
                            out_shape=valid.shape, transform=image.transform,
                            fill=0, dtype="uint8",
                        ).astype(bool)
                        valid &= ~protected_mask
                    count = int(valid.sum())
                    if not count:
                        continue
                    tile_options = dict(options)
                    if method == "slic":
                        target = n_segments
                        if target is None:
                            target = round(count * pixel_area / (math.pi * crown_radius ** 2))
                        tile_options["n_segments"] = max(1, min(count, target))
                    try:
                        # Native masked SLIC cannot initialize seeds from one pixel.
                        if count == 1 and method == "slic":
                            labels = valid.astype(np.int32)
                        else:
                            labels = create_labels(image, method=method, mask=valid, **tile_options)
                        if not np.array_equal(labels > 0, valid):
                            raise RuntimeError("Segmentation did not cover exactly the included pixels; "
                                               "existing objects were kept.")
                        geometries = labels_to_segments(labels, image).geometry
                    except Exception as exc:
                        exc.add_note(f"While segmenting tile at column {col}, row {row} ({method}).")
                        raise
                    store.replace(replaceable, geometries)

        transform, crs = source.transform, source.crs

    coefficients = [transform.a, transform.b, transform.d, transform.e,
                    transform.c, transform.f]
    segments = gpd.GeoDataFrame(
        {"segment_id": np.arange(1, len(store.geometries) + 1, dtype=np.int64)},
        geometry=[affine_transform(geometry, coefficients)
                  for geometry in store.geometries.values()], crs=crs,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".obia-segments-", dir=output_dir) as staging:
        staged_output = Path(staging) / "segments.gpkg"
        segments.to_file(staged_output, layer="segments", driver="GPKG", index=False)
        os.replace(staged_output, output_dir / "segments.gpkg")
