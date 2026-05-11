# OBIA

[![PyPI](https://img.shields.io/pypi/v/obia.svg)](https://pypi.org/project/obia/)
[![PyPI Downloads](https://static.pepy.tech/badge/obia)](https://pepy.tech/projects/obia)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/obia?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/obia)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/obia/tests.yml?branch=main&label=tests)](https://github.com/iosefa/obia/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/iosefa/obia/docs.yml?branch=main&label=docs)](https://github.com/iosefa/obia/actions/workflows/docs.yml)
[![Contributors](https://img.shields.io/github/contributors/iosefa/obia.svg?label=contributors)](https://github.com/iosefa/obia/graphs/contributors)
[![License](https://img.shields.io/github/license/iosefa/obia)](https://github.com/iosefa/obia/blob/main/LICENSE)

**Object-based image analysis tools for geospatial rasters.**

OBIA segments a raster into image objects, summarizes each object with feature columns, and uses those object-level features for classification or review. The main output is a GeoDataFrame of segment polygons that can be saved, labelled, enriched, and classified.

The library supports:

- GeoTIFF loading with Rasterio metadata
- SLIC and quickshift segmentation
- spectral and texture summaries for segment objects
- optional point-cloud height, intensity, and density features
- point-to-segment labelling for training data
- random forest and MLP segment classification
- tiled large-raster workflows for supported segmentation methods

## First Segmentation

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

objects.segments.to_file("segments.gpkg")
```

`objects.segments` is a GeoDataFrame. Each row is a segment polygon with a `segment_id` and calculated feature columns.

Add point-cloud features to the same rows when LiDAR or SfM points are available:

```python
from obia.pointcloud import add_pointcloud_features

segments = add_pointcloud_features(
    objects.segments,
    pointcloud="/path/to/points.laz",
    metrics=["height", "intensity", "density"],
)
```

## Next Steps

- [Installation](installation.md): install OBIA with pip or set up a development environment.
- [Concepts](concepts.md): understand segment objects, feature sources, labels, and classification.
- [Segmentation](usage/segmentation.md): create object polygons and feature tables.
- [Classification](usage/classification.md): label segments and train a classifier.
- [Point Clouds](usage/pointcloud.md): add point-cloud metrics to segment objects.
- [Large Rasters](usage/large-rasters.md): use tiled segmentation utilities.
- [API Reference](api/index.md): inspect generated API documentation.
