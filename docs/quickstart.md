# Quickstart

This quickstart shows the minimal segmentation -> classification flow.

## 1. Load a raster

```python
from obia.handlers.geotif import open_geotiff

img = open_geotiff("/path/to/image.tif")
```

## 2. Segment into objects

```python
from obia.segmentation.segment import segment

segments = segment(
    img,
    segmentation_bands=[0, 1, 2],
    method="slic",
    n_segments=3000,
    compactness=10,
)

# GeoDataFrame with per-segment features
objects_gdf = segments.segments
```

## 3. Train/predict segment classes

```python
from obia.classification.classify import classify

# training_classes must include: feature_class, geometry, segment_id, and feature columns
result = classify(objects_gdf, training_classes, method="rf", n_estimators=300)
classified_gdf = result.classified
```

## 4. Save outputs

```python
objects_gdf.to_file("segments.gpkg")
classified_gdf.to_file("classified_segments.gpkg")
```

## Detection flow

For object detection workflows, use:

- `obia.detection.dataset.TreeDetectionDataset`
- `obia.detection.models.build_detection_model`
- `obia.detection.train.train_model`
- `obia.detection.predict.predict`
