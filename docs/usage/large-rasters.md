# Large Rasters

`segment()` processes an image in memory. When a full scene is too large, use
`create_tiled_segments()` to segment overlapping tiles and merge their polygons.
This workflow uses SLIC and requires GDAL Python bindings.

The [development environment](../installation.md#development-environment)
includes GDAL. You can also install it through conda-forge in your own environment.

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

The function writes intermediate outputs and a merged `segments.gpkg` under
`output_dir`. Tile size and buffer width are measured in pixels.

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

The mask should align with the source raster grid.

## Choose tile size and overlap

Start with tiles that are large enough to preserve the objects you care about but small enough to keep memory use stable.

Try a tile size such as `512` on a representative area, then adjust it for memory
use and object size. A larger source image does not necessarily need larger tiles.

Increase `buffer` when visible tile-edge artifacts appear. Larger buffers reduce boundary artifacts but increase processing cost.

Inspect boundaries near tile edges before processing the full raster. Use
`segment()` when the image fits comfortably in memory and a single set of
boundaries is easier to inspect.
