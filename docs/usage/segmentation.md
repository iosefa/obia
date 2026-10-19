# Segmentation

OBIA wraps scikit-image's region-segmentation algorithms with a common raster,
label-image, polygon, and feature-table interface.

## Load a Raster

```python
from obia.handlers.geotif import open_geotiff
from obia.segmentation.segment import segment

image = open_geotiff("/path/to/image.tif")
```

`open_geotiff()` reads all bands by default. Its `bands` argument uses one-based
Rasterio band numbers. Once loaded, `segmentation_bands` and `statistics_bands`
use zero-based indexes into the loaded image.

## Choose an Algorithm

The `method` names match the functions in
[scikit-image's segmentation API](https://scikit-image.org/docs/stable/api/skimage.segmentation.html).

| Method | Input bands | Additional inputs | Result |
| --- | --- | --- | --- |
| `slic` | One or more | Optional `n_segments`, `compactness`; `slic_zero=True` enables SLICO | Superpixels |
| `quickshift` | One or more | Optional `kernel_size`, `max_dist`, `ratio` | Mode-seeking regions |
| `felzenszwalb` | One or more | Optional `scale`, `sigma`, `min_size` | Graph-based regions |
| `watershed` | Exactly one | Optional `markers` array or count, `compactness` | Basins in the input surface |
| `random_walker` | One or more | Required `labels` seed array | Regions grown from seeds |
| `chan_vese` | Exactly one | Optional `init_level_set`, `max_num_iter` | Foreground objects |
| `morphological_chan_vese` | Exactly one | Required `num_iter`; optional `init_level_set` | Foreground objects |
| `morphological_geodesic_active_contour` | Exactly one | Required `num_iter`; optional `init_level_set`, `balloon` | Foreground objects |
| `active_contour` | One or more | Required closed `snake` coordinates | Pixels inside the fitted contour |
| `flood` | Exactly one | Required `seed_point`; optional `tolerance` | Region connected to the seed |

List supported methods programmatically:

```python
from obia.segmentation import SEGMENTATION_METHODS

print(SEGMENTATION_METHODS)
```

SLIC is the default. Algorithm-specific keyword arguments are passed to
scikit-image, with the input/output adaptations described below. Helpers such as
`find_boundaries`, `clear_border`, and `join_segmentations` are available directly
from scikit-image. `flood_fill` paints the region found by `flood`; use `method="flood"`
to turn that same region-growing algorithm into OBIA objects.

## Superpixels and Graph Segmentation

```python
objects = segment(
    image,
    segmentation_bands=[0, 1, 2],
    statistics_bands=[0, 1, 2, 3],
    method="slic",
    n_segments=3000,
    compactness=10,
)

quickshift_objects = segment(
    image, segmentation_bands=[0, 1, 2], method="quickshift",
    kernel_size=3, max_dist=6, ratio=0.5,
)

felzenszwalb_objects = segment(
    image, segmentation_bands=[0, 1, 2], method="felzenszwalb",
    scale=100, sigma=0.8, min_size=50,
)
```

SLIC controls approximate object count with `n_segments`. Felzenszwalb controls
object size indirectly: larger `scale` generally produces larger regions.
Quickshift's `convert2lab` defaults to True for three selected bands and False
otherwise. For three bands that are not RGB, pass `convert2lab=False`; the same
option is available for SLIC. Channels are always stored last in OBIA.

## Seeded Segmentation

Markers use pixel positions, not map coordinates. Random walker requires an
integer array: positive values identify seed classes, zero means unlabelled,
and negative values mark inactive pixels. Watershed accepts an integer marker
array or a desired basin count; without markers, scikit-image finds local minima.

```python
import numpy as np

height, width = image.img_data.shape[:2]
markers = np.zeros((height, width), dtype=np.int32)
markers[height // 4, width // 4] = 1
markers[3 * height // 4, 3 * width // 4] = 2

watershed_objects = segment(
    image, segmentation_bands=[0], method="watershed", markers=markers,
)

random_walker_objects = segment(
    image, segmentation_bands=[0, 1, 2], method="random_walker",
    labels=markers, beta=130, mode="cg_j",
)
```

Choose seed locations that represent the regions you want. Watershed operates on
the selected band as a surface: low values are flooded first. To segment a
precomputed gradient or other elevation surface, put that surface in a raster
band and select it. `watershed_line=True` leaves separating pixels as background.
Random walker's `mode="cg_mg"` requires the optional upstream dependency `pyamg`.

## Level Sets and Active Contours

```python
chan_vese_objects = segment(
    image, segmentation_bands=[0], method="chan_vese",
    init_level_set="checkerboard", max_num_iter=200,
)

morphological_objects = segment(
    image, segmentation_bands=[0], method="morphological_chan_vese",
    num_iter=100, init_level_set="disk", smoothing=2,
)

geodesic_objects = segment(
    image, segmentation_bands=[0], method="morphological_geodesic_active_contour",
    num_iter=100, init_level_set="disk", balloon=1,
    gradient_kwargs={"alpha": 100, "sigma": 2},
)
```

Morphological geodesic active contours need an edge-stopping image. OBIA computes
`skimage.segmentation.inverse_gaussian_gradient` from the selected band by default.
`gradient_kwargs` controls that preprocessing. Alternatively, pass `gimage=array`
with a finite `(height, width)` edge-stopping image; it is used as supplied and
cannot be combined with `gradient_kwargs`.

`init_level_set` also accepts an array matching the raster's spatial shape.
Scikit-image's `disk_level_set` and `checkerboard_level_set` helpers can create one.

For a parametric active contour, supply initial `(row, column)` pixel coordinates
around the object. Do not duplicate the endpoint of a periodic snake:

```python
angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
snake = np.column_stack((
    height / 2 + height / 4 * np.sin(angle),
    width / 2 + width / 4 * np.cos(angle),
))

contour_objects = segment(
    image, segmentation_bands=[0], method="active_contour",
    snake=snake, alpha=0.01, beta=0.1, max_num_iter=500,
)
```

OBIA fills the fitted closed contour to create objects. Open snakes do not define
an enclosed region, so use scikit-image directly for nonperiodic boundary conditions.

## Flood Region Growing

```python
flood_objects = segment(
    image, segmentation_bands=[0], method="flood",
    seed_point=(height // 2, width // 2), tolerance=0.05,
)
```

The tolerance is in normalized [0, 1] units by default. To use the raster's
original units, pass `normalize=False` and a tolerance in those units.

## Band Values, Masks, and Output

By default, OBIA normalizes each selected band independently to [0, 1]. Constant
bands become zero. The source raster is never modified. `normalize=False` preserves
input values. Algorithms that require a scalar image need one selected band;
OBIA does not silently average RGB or multispectral bands.

`segmentation_bands` controls boundaries. `statistics_bands` controls the bands
summarized afterward, and can include bands that were not used for segmentation.
Feature statistics use the original raster values.

Use `feature_options` to request spectral distributions, indices specified with
numeric band positions, shape measurements, and configurable texture. Alternatively,
run `obia.features.extract_features()` on existing polygons to change measurements
without rerunning segmentation. See [raster object features](raster-features.md)
for configuration, formulas, and examples.

An optional boolean `mask` with shape `(height, width)` restricts output to True
pixels. SLIC, watershed, and random walker use the mask during segmentation.
Other algorithms operate on the full prepared image, then their output is clipped;
the mask is not a barrier to their evolution or region growing. Excluded input
pixels are set to zero in the working copy. Nonfinite values inside the mask are
rejected; nonfinite values outside it are ignored. An entirely empty mask returns
no objects.

The returned `Segments` object stores:

- `objects.labels`: an int32 array with a unique positive ID per connected object.
- `objects._segments`: the same label image, retained for compatibility.
- `objects.segments`: a GeoDataFrame whose `segment_id` values match the label image.
- `objects.method` and `objects.params`: the method and explicitly supplied parameters.

Zero denotes background or excluded pixels. Binary methods polygonize only the
foreground. Superpixel label zero is retained as an object and assigned a positive
ID. Disconnected pieces receive different object IDs, even when an upstream
algorithm gives them the same label. Upstream marker IDs are therefore not class
labels in the output table; use the labelling workflow to attach training classes.

Scikit-image options that change the return type (`return_tree=True`,
`extended_output=True`, `return_full_prob=True`) raise a clear error. Use the
upstream functions for trees, energies, probability maps, or open contour coordinates.

For labels or polygons without feature calculation:

```python
from obia.segmentation import create_labels, create_segments, labels_to_segments

labels = create_labels(image, segmentation_bands=[0], method="watershed", markers=50)
polygons = labels_to_segments(labels, image)
# Or run segmentation and polygonization in one call:
polygons = create_segments(image, segmentation_bands=[0], method="felzenszwalb", scale=100)
```

## Save and Enrich Objects

```python
objects.segments.to_file("segments.gpkg", driver="GPKG")

from obia.pointcloud import add_pointcloud_features

enriched = add_pointcloud_features(
    objects.segments, pointcloud="/path/to/points.laz",
    metrics=["height", "intensity", "density"],
)
```

Inspect object boundaries before training a classifier. The separate
[large-raster tiling workflow](large-rasters.md) still supports SLIC only; its
stitching logic does not yet handle the other methods.
