# OBIA

[![PyPI](https://img.shields.io/pypi/v/obia.svg?release=0.1.4)](https://pypi.org/project/obia/)
[![PyPI Downloads](https://static.pepy.tech/badge/obia)](https://pepy.tech/projects/obia)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/obia?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/obia)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/obia/tests.yml?branch=main&label=tests)](https://github.com/iosefa/obia/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/iosefa/obia/docs.yml?branch=main&label=docs)](https://github.com/iosefa/obia/actions/workflows/docs.yml)
[![Contributors](https://img.shields.io/github/contributors/iosefa/obia.svg?label=contributors)](https://github.com/iosefa/obia/graphs/contributors)
[![License](https://img.shields.io/github/license/iosefa/obia)](https://github.com/iosefa/obia/blob/main/LICENSE)

**Object-based image analysis tools for geospatial rasters.**

OBIA groups raster pixels into objects, measures their properties, and classifies
them using labelled examples. Results are stored in a GeoDataFrame: a table of
polygons and their attributes that you can inspect, save, and use in GIS.

With OBIA, you can:

- Load GeoTIFFs with their coordinate reference system and spatial transform.
- Segment imagery into object polygons.
- Measure each object's band statistics, spectral indices, shape, and texture.
- Transfer point labels to objects and train a classifier.
- Train on labelled point pixels and produce a raster class map.
- Evaluate models with grouped or spatial splits, buffers, and cross-validation.
- Add point-cloud measurements or process large rasters in tiles when needed.

## Get started

Follow the [installation guide](installation.md) to set up OBIA. The
[example notebooks](examples/index.md) walk through segmentation, feature
extraction, and classification with runnable code and plots.

## Segment a raster

This example uses the bundled Chesapeake imagery to create polygons and measure
their raster values. Replace the sample path with your GeoTIFF for your own data.

```python
from obia.datasets import chesapeake_sample
from obia.handlers.geotif import open_geotiff
from obia.segmentation.segment import segment

image = open_geotiff(chesapeake_sample("train")["image"])

objects = segment(
    image,
    segmentation_bands=[0, 1, 2],
    statistics_bands=[0, 1, 2, 3],
    method="slic",
    n_segments=600,
    compactness=0.3,
    convert2lab=False,
)

objects.segments.to_file("segments.gpkg", driver="GPKG")
image.rasterio_obj.close()
```

Each row of `objects.segments` contains a polygon, a `segment_id`, and feature
columns. Band indices start at zero: `0` refers to the first loaded band.
`segmentation_bands` determines the boundaries; `statistics_bands` determines
which bands are measured within each object.

## Continue the workflow

- [Concepts](concepts.md): understand objects, features, labels, and predictions.
- [Segmentation](usage/segmentation.md): choose an algorithm and adjust boundaries.
- [Raster features](usage/raster-features.md): add or change object measurements.
- [Classification](usage/classification.md): label objects, fit a model, and save predictions.
- [Pixel classification](usage/pixel-classification.md): learn from point samples and classify raster pixels.
- [Training and validation](usage/validation.md): choose held-out data and compare models.

For other workflows, see [point-cloud features](usage/pointcloud.md),
[large rasters](usage/large-rasters.md), and
[experimental detection](usage/detection.md). The [API reference](api/index.md)
documents function arguments and return values.
