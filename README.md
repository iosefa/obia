# OBIA

Object-based image analysis tools for geospatial rasters.

OBIA groups raster pixels into spatial objects, calculates object-level features, and supports downstream segment classification. The main output is a GeoDataFrame or GeoPackage of segment polygons with feature columns and predicted classes.

Current scope:

- GeoTIFF loading with Rasterio metadata preserved
- SLIC and quickshift segmentation
- segment-level spectral and texture features
- optional point-cloud features joined to the same segment objects
- point-to-segment labelling utilities
- random forest and MLP segment classification
- tiled SLIC utilities for large rasters
- experimental RetinaNet-based detection modules

## Install

### From PyPI

```bash
pip install obia
```

### Point-cloud support

Point-cloud workflows need PDAL. Install PDAL with conda-forge first, then install OBIA with `pip`:

```bash
conda create -n obia-pointcloud -c conda-forge python=3.11 pdal python-pdal
conda activate obia-pointcloud
pip install obia
```

### Developer install

```bash
git clone https://github.com/iosefa/obia.git
cd obia
conda env create -f environment.yml
conda activate obia
```

The development environment installs the package in editable mode with documentation and optional workflow dependencies.

### Docker

Build the image from the repository root:

```bash
docker build -t obia .
```

Release images are published to Docker Hub as `iosefa/obia`:

```bash
docker pull iosefa/obia:latest
```

The image includes OBIA, PDAL, the PDAL Python bindings, and `pyforestscan`.

## Basic Segmentation

```python
from obia.handlers.geotif import open_geotiff
from obia.segmentation.segment import segment

image = open_geotiff("/path/to/image.tif")

objects = segment(
    image,
    segmentation_bands=[0, 1, 2],
    statistics_bands=[0, 1, 2, 3],
    method="slic",
    n_segments=3000,
    compactness=10,
)

segments = objects.segments
segments.to_file("segments.gpkg", driver="GPKG")
```

## Point-Cloud Features

```python
from obia.pointcloud import add_pointcloud_features

segments = add_pointcloud_features(
    segments,
    pointcloud="/path/to/points.laz",
    metrics=["height", "intensity", "density"],
)
```

Point-cloud features are appended as `pc_*` columns, so the enriched segment table can be labelled and classified with the raster-derived features.

## Segment Classification

```python
import geopandas as gpd

from obia.classification.classify import classify
from obia.utils.utils import label_segments

labelled_points = gpd.read_file("/path/to/labelled_points.gpkg")
training_segments, mixed_segments = label_segments(
    segments,
    labelled_points,
)

result = classify(
    segments,
    training_segments,
    method="rf",
    n_estimators=300,
    random_state=42,
)

result.classified.to_file("classified_segments.gpkg", driver="GPKG")
```

## Documentation

Preview the docs locally:

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000`. If the port is already in use:

```bash
mkdocs serve -a 127.0.0.1:8001
```

Documentation source lives in `docs/`.

## License

MIT (`LICENSE`)
