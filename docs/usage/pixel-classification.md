# Pixel classification

Train a classifier from raster values at labelled points, then predict a class
for each valid pixel. Select bands by their zero-based positions in the loaded
image. The result keeps the raster's dimensions, coordinate reference system,
and affine transform.

## Train and classify a raster

Points need a CRS and a `class` column containing string labels or numeric class
codes. OBIA reprojects points to the raster CRS and reads the pixel containing
each point, without interpolation.

```python
import geopandas as gpd

from obia.classification import classify_pixels
from obia.handlers.geotif import open_geotiff

image = open_geotiff("image.tif")
points = gpd.read_file("labelled_points.gpkg")

result = classify_pixels(
    image,
    points,
    bands=[0, 1, 2, 3],
    method="rf",
    n_estimators=200,
    random_state=42,
    n_jobs=1,
    test_size=None,
)
result.write_geotiff("pixel_classes.tif")
print(result.class_mapping)
image.rasterio_obj.close()
```

This example fits all sampled pixels with `test_size=None`. The default reserves
20% of samples for a stratified random holdout. For predictions in new areas,
choose a spatial or grouped split before fitting, or evaluate in a separate area.
The [pixel classification notebook](../examples/04_pixel_classification.ipynb)
provides a complete example using the bundled Chesapeake data.

## Choose a classifier

Pixel and object classification accept the same method names, estimator
parameters, and custom pipelines. For example, replace the model settings above
with `method="xgboost", n_estimators=200, max_depth=4, n_jobs=1`.
See the [classifier table](classification.md#choose-a-classifier) for the available
methods and their input requirements.

Band values are used as loaded. `scale_features="auto"` applies scaling for the
same estimator families as object classification, fitted only on training rows.
A supplied pipeline owns its preprocessing. Classifiers without probabilities,
such as `LinearSVC`, still produce class predictions.

## Inspect the sampled pixels

`sample_pixels()` exposes the training table before fitting:

```python
from obia.classification import sample_pixels

samples = sample_pixels(image, points, bands=[0, 1, 2, 3])
print(samples[["pixel_row", "pixel_col", "b0", "b3", "feature_class"]].head())
print(samples.attrs["obia_sampling"])
```

Each row contains selected band values as `b0`, `b1`, and so on, plus the target
`feature_class` and zero-based raster row and column. Geometry is the pixel centre
in the raster CRS. Other point attributes and indices are retained; generated
columns replace existing columns with the same names. Only the selected band
columns become model inputs.

- Use `label_column="land_cover"` if the point labels have another column name.
- Points in the same pixel count once. Matching labels keep the first point's
  attributes; conflicting labels raise an error.
- Outside points and points on NoData, nonfinite, or masked pixels raise an error.
  Use `invalid_samples="drop"` to discard them explicitly and inspect the counts.
- Source masks and an optional boolean `mask` apply to the selected bands. Invalid
  values in unselected bands do not exclude a pixel.

## Use spatial or grouped validation

The split helpers also accept sampled point tables. Create a split from
`sample_pixels()` and use the same image, points, bands, and sampling options when
classifying. This keeps duplicate removal and row assignments consistent.

```python
from obia.classification import split_objects

partition = split_objects(
    samples,
    method="spatial",
    block_size=500,
    buffer_distance=100,
    test_size=0.2,
    random_state=42,
)
result = classify_pixels(
    image, points, bands=[0, 1, 2, 3], method="rf",
    split=partition, compute_reports=True,
    n_estimators=200, random_state=42, n_jobs=1,
)
print(result.report)
print(result.split.class_counts)
```

Blocks and buffers use the projected CRS's units. Choose distances for your data
and prediction area; the values above are examples. For site or field IDs retained
in `samples`, use `method="grouped", groups="site_id"`. Explicit partitions and
folds from `cross_validation_splits()` also work. See
[training and validation](validation.md) for design choices and class coverage.

`compute_reports=True` requires held-out samples. Reports describe the fitted
model's holdout predictions. OBIA does not automatically refit on those samples.

## Read and save predictions

The result is a `ClassifiedRaster`:

- `classified`: a `uint32` array with the raster's height and width. Zero is
  NoData; codes `1..K` identify the classes in `classes` order.
- `class_mapping`: maps each positive raster code to its original string or
  numeric label. Codes can differ from the input labels, including a label of `0`.
- `prediction_margin`: the gap between the two largest class probabilities.
  Invalid pixels and models without probabilities have `NaN` margins.
- `model`, `bands`, and `feature_columns`: the fitted pipeline and its input order.
- `samples`, `split`, `confusion_matrix`, and `report`: sampling and evaluation
  results, when available.

Set `return_probabilities=True` to retain a `(height, width, classes)` probability
array. Columns follow `result.classes`; invalid pixels are `NaN`. Without
`predict_proba`, `result.probabilities` is `None`. Margins do not measure calibrated
accuracy.

`write_geotiff()` saves the class-code raster with NoData `0`, the source grid,
and a JSON `class_mapping` dataset tag. Read it with Rasterio:

```python
import json
import rasterio

with rasterio.open("pixel_classes.tif") as src:
    codes = src.read(1, masked=True)
    mapping = json.loads(src.tags()["class_mapping"])
```

## Predict another image

```python
other = open_geotiff("another_image.tif")
prediction = result.predict(other, batch_size=65536)
prediction.write_geotiff("another_pixel_map.tif")
other.rasterio_obj.close()
```

This reuses the fitted model and band indices. The new image can have a different
size, transform, or CRS, but its band meanings, order, and value units must match.
No fitting takes place. The new result contains predictions without copying the
training table or earlier holdout report.

`batch_size` limits temporary prediction feature tables. The loaded image and
output arrays remain in memory; requesting probabilities adds one full array
layer per class. The source raster is not changed.
