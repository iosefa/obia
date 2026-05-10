# Large Rasters

The standard `segment()` workflow loads and processes an image in memory. For large rasters, use tiled processing so each tile can be segmented separately.

The tiled path is currently focused on SLIC segmentation and requires GDAL Python bindings.

## Tiled SLIC Segmentation

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

The function writes intermediate tile outputs and a merged segment file under `output_dir`.

## Masked Processing

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

## Choosing Tile Size

Start with tiles that are large enough to preserve the objects you care about but small enough to keep memory use stable.

Practical starting points:

| Raster size | Starting tile size |
| --- | --- |
| medium scenes | `512` |
| large scenes | `512` or `1024` |
| memory-constrained runs | `256` |

Increase `buffer` when visible tile-edge artifacts appear. Larger buffers reduce boundary artifacts but increase processing cost.

## When To Use This Path

Use tiled segmentation when:

- the standard segmentation path exhausts memory
- a full-scene label image would be too large
- you need to process many scenes with the same settings

Use standard `segment()` when possible. It is simpler to inspect, debug, and reproduce.
