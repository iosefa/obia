# Workflows

## A. Segmentation and object features

1. Open raster with `open_geotiff`.
2. Run `segment(...)` with `slic` or `quickshift`.
3. Use returned `Segments.segments` GeoDataFrame for downstream tasks.

Primary modules:

- `obia.handlers.geotif`
- `obia.segmentation.segment`
- `obia.segmentation.segment_boundaries`
- `obia.segmentation.segment_statistics`

## B. Segment classification

1. Build a labeled training table from segments.
2. Run `classify(...)` with `method="rf"` or `method="mlp"`.
3. Export the classified GeoDataFrame.

Primary module:

- `obia.classification.classify`

## C. Detection model training/inference

1. Build dataset with `TreeDetectionDataset` and JSON annotations.
2. Construct RetinaNet with `build_detection_model`.
3. Train with `train_model`.
4. Run inference with `predict`.

Primary modules:

- `obia.detection.dataset`
- `obia.detection.models`
- `obia.detection.train`
- `obia.detection.predict`

## D. Utility pipelines

- tile generation: `obia.utils.tiling`, `obia.utils.training`
- seed generation: `obia.utils.seeds`
- cost-surface generation: `obia.utils.cost`
