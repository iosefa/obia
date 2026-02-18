# obia

`obia` is a Python library for object-based image analysis.

> Note: `obia` is under active development.

## Installation

```bash
conda env create -f environment.yml
conda activate obia
```

## Supported Algorithms

Segmentation:

- `slic`
- `quickshift`

Classification:

- `rf` (Random Forest)
- `mlp` (Multi-layer Perceptron)

## Notable Features

- Large-raster tiled segmentation for `slic` via `obia.utils.tiling.create_tiled_segments`
- Seam-oriented tile handling for `slic` using overlap/buffer logic to reduce boundary artifacts
- End-to-end segmentation + classification workflow using geospatial vector outputs (`GeoPackage`)

## Quickstart (Segmentation + Classification)

```python
from obia.handlers.geotif import open_geotiff
from obia.segmentation.segment import segment
from obia.classification.classify import classify
from obia.utils.utils import label_segments
import geopandas as gpd

img = open_geotiff("/path/to/image.tif")
segments = segment(
    img,
    segmentation_bands=[0, 1, 2],
    method="slic",
    n_segments=3000,
    compactness=10,
)

# Build training labels from points (must have a `class` column + geometry)
labelled_points = gpd.read_file("/path/to/labelled_points.gpkg")
training_classes, mixed_segments = label_segments(segments.segments, labelled_points)
# training_classes now contains segment features + `feature_class`

result = classify(
    segments.segments,
    training_classes,
    method="rf",
    n_estimators=300,
)

segments.segments.to_file("segments.gpkg")
training_classes.to_file("training_classes.gpkg")
result.classified.to_file("classified_segments.gpkg")
```

## Documentation

- docs source: `docs/`
- example notebook: `docs/examples/segmentation-quickstart.ipynb`

## License

MIT (`LICENSE`)
