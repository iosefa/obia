# Large Rasters

`segment()` processes an image in memory. When a full scene is too large, use
`create_tiled_segments()` to segment overlapping tiles and merge their polygons.
Both SLIC and Quickshift are supported. Windowed reads use Rasterio, which is
already included in the core installation; separate GDAL Python bindings are
not required.

## Segment a raster in tiles

```python
from obia.utils.tiling import create_tiled_segments

create_tiled_segments(
    input_raster="/path/to/large_image.tif",
    output_dir="output_tiles",
    method="slic",
    tile_size=512,
    buffer=64,
    n_segments=1000,
    compactness=10,
)
```

The function writes a merged `segments.gpkg` under `output_dir`, with the source
CRS and a unique `segment_id` for each connected object. Tile size and buffer
width are measured in pixels. An existing output is replaced only after the
complete new output has been written.

For SLIC, an explicit `n_segments` is an approximate target for each window,
capped at its valid pixel count. Without it, `crown_radius` (default `5`, in map
units) estimates the count from the included area. For geographic rasters, supply
`n_segments` explicitly or reproject to a suitable projected CRS first.

## Tiled Quickshift

Use the same function with Quickshift's parameters:

```python
create_tiled_segments(
    input_raster="/path/to/large_image.tif",
    output_dir="quickshift_tiles",
    method="quickshift",
    segmentation_bands=[0, 1, 2],
    tile_size=512,
    buffer=64,
    kernel_size=3,
    max_dist=6,
    ratio=0.5,
    convert2lab=False,
)
```

Band indexes are zero-based. Select bands that exist in your raster. For RGB
imagery, `convert2lab=True` is also supported; use False for other band combinations.
Quickshift does not take `n_segments`, and `crown_radius` does not apply to it.

### How overlaps are handled

The first pass segments the black squares of a checkerboard. The second pass
segments the white squares with an overlapping buffer:

1. Existing objects entirely inside the buffered window become candidates for
   replacement, so new objects can extend across the original tile boundaries.
2. Objects crossing the buffered window's outer edge are kept whole. Their pixels
   are excluded from the new segmentation with a boolean mask.
3. The new labels must cover exactly the included pixels before any existing
   objects are removed. Segmentation errors stop the run instead of silently
   dropping a tile.

This also protects objects from earlier white tiles where their buffers overlap.
Objects at the outside edge of the source raster are retained. A buffer too narrow
to contain an edge object leaves that object intact; it is never discarded merely
because it touches a tile boundary.

SLIC uses its native mask. Quickshift has no native mask, so OBIA fills excluded
pixels with a distinct finite value after normalizing the valid data and converting
colour space. If `sigma` requests smoothing, only valid pixels contribute to the
smoothed values, and the fill is introduced afterward.

Quickshift may still give real and excluded pixels the same label, especially
when `ratio=0` ignores colour differences. OBIA clips **every** resulting label
using the saved boolean mask, preserving its valid portion. Only labels with no
valid pixels disappear, and disconnected remnants receive separate IDs. The
fill value itself is never used to decide which pixels to delete.

## Apply a mask

Pass a binary mask raster to avoid segmenting invalid or out-of-scope areas:

```python
create_tiled_segments(
    input_raster="/path/to/large_image.tif",
    input_mask="/path/to/mask.tif",
    output_dir="output_tiles",
    method="slic",
    tile_size=512,
    buffer=64,
    n_segments=1000,
    compactness=10,
)
```

The mask must have the same dimensions, transform, and CRS as the source raster;
misaligned masks raise an error. Nonzero, finite, non-NoData mask pixels are
included. Source NoData, source raster masks, and nonfinite values in selected
bands are also excluded, whether or not `input_mask` is supplied. An all-excluded
scene produces an empty GeoPackage layer.

## Choose tile size and overlap

Start with tiles that are large enough to preserve the objects you care about but small enough to keep memory use stable.

Try a tile size such as `512` on a representative area, then adjust it for memory
use and object size. A larger source image does not necessarily need larger tiles.

Increase `buffer` when visible tile-edge artifacts appear. It must be a nonnegative
integer smaller than `tile_size`; zero disables overlap. Larger buffers give
segmentation more context and allow larger edge objects to be replaced, but
increase processing cost.

Both passes currently run sequentially. The white pass depends on accepted objects
from earlier windows. Raster working memory is limited to the selected bands in
one buffered tile, while accepted polygons accumulate in memory as the run proceeds.

Overlap replacement can avoid straight cuts at the original tile boundaries, but
does not guarantee the same segmentation as a whole-image run or invisible seams.
Each window has limited context and, by default, its own band normalization.
Quickshift's artificial fill can also influence nearby boundaries. Clipping
preserves valid coverage; it does not make the fill a native Quickshift mask.

Inspect boundaries near tile edges before processing the full raster. Use
`segment()` when the image fits comfortably in memory and a single set of
boundaries is easier to inspect.
