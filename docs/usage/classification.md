# Classification

Classification assigns class labels to segment objects. The classifier uses segment-level features rather than raw pixels.

Raster-derived features and point-cloud-derived features can be used together. The classifier only requires one segment table with numeric feature columns.

## Prepare Training Labels

Start with:

- a segment GeoDataFrame from `segment(...).segments`
- a labelled point GeoDataFrame with a `class` column

If point-cloud features are needed, add them before labelling so the labelled training segments and prediction segments have the same feature columns.

```python
import geopandas as gpd

from obia.utils.utils import label_segments

segments_gdf = gpd.read_file("segments.gpkg")
labelled_points = gpd.read_file("labelled_points.gpkg")

training_segments, mixed_segments = label_segments(
    segments_gdf,
    labelled_points,
)
```

`training_segments` contains a `feature_class` column. `mixed_segments` contains segment IDs that intersected more than one class.

Review mixed segments before training. They often indicate label noise, poor segmentation, or sample points near class boundaries.

## Train and Predict

```python
from obia.classification.classify import classify

result = classify(
    segments_gdf,
    training_segments,
    method="rf",
    n_estimators=300,
    random_state=42,
)

classified = result.classified
```

The classified GeoDataFrame includes:

- `predicted_class`
- `prediction_margin`
- original segment geometry
- original feature columns

## Random Forest

Random forest is the default starting point:

```python
result = classify(
    segments_gdf,
    training_segments,
    method="rf",
    n_estimators=300,
    max_depth=None,
    random_state=42,
)
```

Use it before trying neural-network classification. It is usually easier to diagnose on small tabular feature sets.

## MLP

Use `method="mlp"` for scikit-learn's multi-layer perceptron classifier:

```python
result = classify(
    segments_gdf,
    training_segments,
    method="mlp",
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42,
)
```

MLP classification is more sensitive to feature scaling, class balance, and training size.

## Reports and Explanations

Set `compute_reports=True` to calculate a confusion matrix and classification report on the train/test split:

```python
result = classify(
    segments_gdf,
    training_segments,
    method="rf",
    compute_reports=True,
    n_estimators=300,
)

print(result.confusion_matrix)
print(result.report)
```

SHAP explanations require the optional `explain` dependency group:

```bash
pip install "obia[explain]"
```

Then run:

```python
result = classify(
    segments_gdf,
    training_segments,
    method="rf",
    compute_shap=True,
    n_estimators=300,
)
```

## Save Output

```python
classified.to_file("classified_segments.gpkg", driver="GPKG")
```

The output remains a vector object map. If a raster classification map is required, rasterize the classified polygons onto the source raster grid as a separate post-processing step.
