# Concepts

Object-based image analysis treats groups of pixels as analysis units. Instead of classifying every pixel independently, OBIA first partitions an image into spatial objects and then summarizes each object with features such as band means, variance, texture, and shape-related measurements.

This is useful when the target classes are better represented by patches, crowns, fields, canopy gaps, or other spatial units than by individual pixels.

## Image Representation

`obia.handlers.geotif.open_geotiff` reads a raster into an `Image` object. The object stores:

- `img_data`: raster values as a NumPy array with shape `(height, width, bands)`
- `crs`: the raster coordinate reference system
- `transform`: the Rasterio affine transform
- `rasterio_obj`: the source Rasterio dataset

Band indexes in OBIA are zero-based after loading. If a GeoTIFF has bands 1, 2, and 3, they are referenced in OBIA as `0`, `1`, and `2`.

## Segments And Objects

Segmentation converts raster pixels into polygon geometries. The public `segment()` function runs two steps:

1. `create_segments()` produces segment polygons.
2. `create_objects()` calculates segment-level statistics.

The result is a `Segments` object. The main table is `Segments.segments`, a GeoDataFrame with one row per segment.

The supported segmentation methods are:

- `slic`
- `quickshift`

Both methods come from scikit-image. OBIA passes additional keyword arguments through to the selected algorithm.

## Feature Sources

The classification workflow expects a segment table with numeric feature columns. Geometry and identifier columns are kept for geospatial output, but the classifier is trained on the remaining feature columns.

Common feature groups include:

- spectral statistics from selected raster bands
- texture statistics from gray-level co-occurrence calculations
- point-cloud height, intensity, and density metrics
- segment identifiers and geometry for output

Raster features are created during segmentation. Point-cloud features are added later as an enrichment step, using the same segment polygons.

Feature calculation can be expensive on large rasters or dense point clouds. Start with a small representative area before scaling up to a full scene.

## Labels

`label_segments()` joins labelled points to segment polygons. A segment receives a class when all intersecting label points agree. Segments intersecting multiple classes are returned separately as mixed segments so they can be inspected or excluded.

The labelled point table must contain:

- `geometry`
- `class`

The labelled segment table produced by `label_segments()` contains `feature_class`, which is the target column expected by `classify()`.

## Classification

`classify()` supports:

- `method="rf"` for random forests
- `method="mlp"` for multi-layer perceptrons

The function returns a `ClassifiedImage` object. Its `classified` attribute is the classified GeoDataFrame with predicted classes and prediction margins.

SHAP explanations are optional. Install the `explain` extra and pass `compute_shap=True` when feature-attribution output is needed.

## Detection

The experimental `obia.detection` package contains a RetinaNet-based training and inference path for raster object detection. It is separate from the segmentation/classification workflow and depends on PyTorch, Torchvision, Albumentations, and Matplotlib.

Use the detection module when the desired output is bounding boxes rather than segment polygons.

## Large Rasters

Large rasters should be processed in smaller pieces. `obia.utils.tiling.create_tiled_segments()` runs SLIC segmentation over tiles and merges the resulting polygons. This path requires GDAL Python bindings and should be treated as an advanced workflow.

Use the standard `segment()` path first. Move to tiled processing when memory use or runtime becomes impractical.

## Point Clouds

Point-cloud feature extraction is an optional segment enrichment step. Raster segmentation creates the object geometry and raster feature table first. Point-cloud metrics are then calculated inside each segment polygon and appended as `pc_*` columns.

This keeps the workflow simple: one segment table, multiple feature sources.

Point-cloud path inputs require optional dependencies such as PDAL or pyforestscan. In-memory point clouds can be passed as DataFrames, GeoDataFrames, or NumPy arrays.

Typical point-cloud columns include:

- `pc_point_count`
- `pc_density`
- `pc_z_mean`
- `pc_z_p95`
- `pc_intensity_mean`

## Practical Limits

OBIA does not solve every remote-sensing classification problem. It works best when:

- raster metadata is correct
- class boundaries are reasonably represented by image segments
- segmentation bands contain the structure needed to form useful objects
- training labels are spatially accurate
- classes have enough examples for supervised learning

Poor segmentation usually leads to poor classification. Tune segmentation parameters before investing heavily in classifier settings.
