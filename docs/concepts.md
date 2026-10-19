# Concepts

Object-based image analysis uses regions of an image as the units of analysis.
A region might represent a field, tree crown, canopy gap, or patch of land cover.
Segmentation defines these regions; their measurements become inputs to a
classifier.

The raster workflow is:

1. **Segment the image** into objects and convert their boundaries to polygons.
2. **Measure the objects**, using band values, spectral indices, shape, or texture.
3. **Label a sample of objects** and choose training and test subsets.
4. **Train and evaluate a classifier**, then predict classes for the object map.

The same polygon table carries geometry, measurements, labels, and predictions
through these steps. You can save it between steps or start with polygons you
already have. The [example notebooks](examples/index.md) demonstrate the workflow.

You can also classify individual pixels. `classify_pixels()` reads band values at
labelled points, trains a classifier, and predicts a class raster.
Its `ClassifiedRaster` result stores class codes, their original labels, the fitted
model, and raster metadata. See [pixel classification](usage/pixel-classification.md).

## Raster data and band indices

`obia.handlers.geotif.open_geotiff` returns an instance of OBIA's own
[`Image` class](api/handlers.md#obia.handlers.geotif.Image). It stores:

- `img_data`: raster values as a NumPy array with shape `(height, width, bands)`
- `crs`: the raster coordinate reference system
- `transform`: the Rasterio affine transform
- `rasterio_obj`: the source Rasterio dataset
- `valid_mask`: which pixels are valid in each loaded band

`image.to_image([0, 1, 2])` creates a Pillow image for display.

Band indices are zero-based after loading: `0` is the first loaded band. The
loader's optional `bands` argument uses one-based file band numbers. For example,
after `open_geotiff(path, bands=[4, 1, 2])`, OBIA indices `0`, `1`, and `2` refer
to file bands 4, 1, and 2, respectively.

## Segments and objects

Segmentation groups pixels into connected regions. Each region becomes an object
with a polygon boundary and a unique `segment_id`. The `segment()` function
combines three operations:

1. `create_labels()` creates a raster of integer object IDs.
2. `labels_to_segments()` converts the regions to georeferenced polygons.
3. `create_objects()` adds measurements to the polygon table.

`segment()` returns a `Segments` result. Its `.segments` attribute is the
GeoDataFrame, and `.labels` is the object-ID raster. If you only need polygons,
`create_segments()` combines the first two operations and returns a GeoDataFrame
directly. Use `extract_features()` to add measurements later.

The [segmentation guide](usage/segmentation.md) explains how to choose an algorithm
and adjust its inputs.

Each connected object has a positive label ID matching its polygon's
`segment_id`; zero is background or excluded data. Binary methods create objects
from foreground pixels. Segmentation works on a copy of the selected bands, so
the original raster values remain available for feature calculation.

## Object features

Features are numeric measurements used to distinguish classes. The classifier
learns from selected feature columns; geometry and object IDs remain in the
table for mapping and export.

Common feature groups include:

- spectral distributions from selected raster bands
- spectral indices calculated from explicit numeric band positions
- polygon size and shape measurements
- texture statistics from masked gray-level co-occurrence matrices and local entropy
- optional point-cloud height, intensity, and density measurements

Raster features can be created during segmentation or recalculated independently
with `obia.features.extract_features()`, using the same polygons. See
[raster object features](usage/raster-features.md) for selectable measurements,
formulas, and units. Point-cloud features are added later as an enrichment step,
using the same segment polygons.

Start with a small representative area when choosing features. Texture and
point-cloud measurements can be expensive over a full scene.

## Labels

Training labels identify known classes, such as forest, water, or soil.
`label_segments()` transfers labels from points to polygons. An object receives
a class when its intersecting points agree. The function also returns the IDs
of objects with conflicting labels, so you can review them.

The point table needs a `geometry` column and a `class` column, with coordinates
in the same CRS as the objects.

The returned labelled objects have a `feature_class` column, which is the target
expected by `classify()`. These class labels are separate from the numeric
`segment_id` values used to identify objects.

## Classification

`classify()` fits a model to labelled objects and predicts a class for each object
in the prediction table. Choose a method such as random forest, logistic
regression, or support vector machines, or supply a compatible estimator or
pipeline. See the [classification guide](usage/classification.md) for model
choices and preprocessing.

The result retains the fitted model and feature order. Its `.classified`
attribute contains the object polygons with `predicted_class` and
`prediction_margin` columns. The margin is the gap between the two largest class
probabilities; it is missing for models without probability estimates.

## Training and validation

Training objects fit the model. Held-out objects measure how well it predicts
labels it has not seen during fitting. For predictions in new fields or sites,
keep those units separate when splitting data. Nearby objects can share enough
information to make a random split look overly optimistic.

Use `split_objects()` to define and inspect a partition, then pass it to
`classify(split=...)`. For model selection, `cross_validation_splits()` provides
reusable folds. The [validation guide](usage/validation.md) explains spatial
blocks, buffers, explicit test sets, and final evaluation.

## Other workflows

- [Point-cloud features](usage/pointcloud.md) add height, intensity, and density
  measurements to the same polygon table as `pc_*` columns.
- [Large rasters](usage/large-rasters.md) covers tiled SLIC processing when a
  full image is impractical to process in memory.
- [Experimental detection](usage/detection.md) produces bounding boxes with a
  separate model-training workflow.

## Check the inputs

Results depend on the quality of the objects, features, and labels. Check that:

- raster metadata is correct
- class boundaries are reasonably represented by image segments
- segmentation bands contain the structure needed to form useful objects
- training labels are spatially accurate
- classes have enough examples for supervised learning

Inspect object boundaries and class coverage before comparing classifiers.

## Further reading

For background on object-based and geographic object-based image analysis:

- Blaschke, T. (2010). Object based image analysis for remote sensing. *ISPRS Journal of Photogrammetry and Remote Sensing*, 65(1), 2-16. <https://doi.org/10.1016/j.isprsjprs.2009.06.004>
- Hay, G. J., & Castilla, G. (2008). Geographic Object-Based Image Analysis (GEOBIA): A new name for a new discipline. In T. Blaschke, S. Lang, & G. J. Hay (Eds.), *Object-Based Image Analysis: Spatial Concepts for Knowledge-Driven Remote Sensing Applications* (pp. 75-89). Springer. <https://doi.org/10.1007/978-3-540-77058-9_4>

For the segmentation algorithms exposed by OBIA:

- Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., & Suesstrunk, S. (2012). SLIC superpixels compared to state-of-the-art superpixel methods. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 34(11), 2274-2282. <https://doi.org/10.1109/TPAMI.2012.120>
- Vedaldi, A., & Soatto, S. (2008). Quick shift and kernel methods for mode seeking. In *Computer Vision - ECCV 2008* (pp. 705-718). Springer. <https://doi.org/10.1007/978-3-540-88693-8_52>
