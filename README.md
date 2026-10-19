# OBIA

[![PyPI](https://img.shields.io/pypi/v/obia.svg?release=0.1.4)](https://pypi.org/project/obia/)
[![PyPI Downloads](https://static.pepy.tech/badge/obia)](https://pepy.tech/projects/obia)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/obia?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/obia)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/obia/tests.yml?branch=main&label=tests)](https://github.com/iosefa/obia/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/iosefa/obia/docs.yml?branch=main&label=docs)](https://github.com/iosefa/obia/actions/workflows/docs.yml)
[![Contributors](https://img.shields.io/github/contributors/iosefa/obia.svg?label=contributors)](https://github.com/iosefa/obia/graphs/contributors)
[![License](https://img.shields.io/github/license/iosefa/obia)](https://github.com/iosefa/obia/blob/main/LICENSE)

Object-based image analysis tools for geospatial rasters.

OBIA groups raster pixels into spatial objects, calculates object-level features, and supports downstream segment classification. The main output is a GeoDataFrame or GeoPackage of segment polygons with feature columns, optional point-cloud metrics, and predicted classes.

Current scope:

- GeoTIFF loading with Rasterio metadata preserved
- ten scikit-image region-segmentation methods, including SLIC, quickshift,
  Felzenszwalb, watershed, random walker, active contours, and flood region growing
- segment-level spectral distributions and indices, shape features, and configurable texture
- optional point-cloud features joined to the same segment objects
- point-to-segment labelling utilities
- scikit-learn object classifiers and pipelines, plus optional XGBoost
- tiled SLIC utilities for large rasters
- experimental RetinaNet-based detection modules

## Install

### From PyPI

```bash
pip install obia
```

### Point-cloud support

Point-cloud workflows need PDAL. Install PDAL with conda-forge first, then install OBIA with `pip` inside that environment:

```bash
conda create -n obia-pointcloud -c conda-forge python=3.11 pdal python-pdal
conda activate obia-pointcloud
pip install obia
```

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

### Developer install

```bash
git clone https://github.com/iosefa/obia.git
cd obia
conda env create -f environment.yml
conda activate obia
```

The development environment installs the package in editable mode with documentation and optional workflow dependencies.

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

Segmentation also accepts `method="felzenszwalb"`, `"watershed"`,
`"random_walker"`, `"chan_vese"`, `"morphological_chan_vese"`,
`"morphological_geodesic_active_contour"`, `"active_contour"`, and `"flood"`.
Scalar methods require one selected band; seeded methods require markers or an
initial contour. See the [segmentation guide](https://obia.sefa.ai/usage/segmentation/)
for inputs, parameters, and examples for each method.

## Raster Object Features

Calculate additional features on existing polygons without repeating segmentation.
All bands use zero-based numeric positions in the loaded image:

```python
from obia.features import TextureConfig, extract_features

segments = extract_features(
    segments, image, bands=[0, 1, 2, 3],
    spectral=("mean", "std", "median", "iqr", "mad", "valid_count", "valid_fraction"),
    percentiles=(10, 90),
    indices={"ndvi": (3, 0)},
    shape=True,
    texture=TextureConfig(bands=(0,), distances=(1, 2), entropy_radii=(1, 3)),
)
```

Choose index inputs to match your raster's band order. Shape uses projected map
units, or `shape_units="pixel"` for pixel measurements. See the
[raster feature guide](https://obia.sefa.ai/usage/raster-features/) for definitions,
masks, and texture settings.

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

Choose from 39 named methods, including `"extra_trees"`, `"svc"`, `"knn"`,
`"logistic_regression"`, `"hist_gradient_boosting"`, `"gaussian_nb"`, `"lda"`,
and `"xgboost"`, or pass a scikit-learn classifier or pipeline as `method`.
Install `pip install "obia[xgboost]"` for XGBoost. Feature scaling is fitted on
training objects and reused for predictions; classifiers without probabilities
return a missing prediction margin. See the
[classification guide](https://obia.sefa.ai/usage/classification/) for the full
method list, custom pipelines, class constraints, and evaluation.

## Example Notebooks

Three small notebooks generate their own synthetic raster and run independently:

- [Raster segmentation](docs/examples/01_segmentation.ipynb): compare algorithms,
  create object polygons, and export segment IDs and a GeoPackage.
- [Raster object features](docs/examples/02_raster_features.ipynb): spectral
  summaries, indices, shape, texture, and quality masks.
- [Object classification](docs/examples/03_classification.ipynb): label objects
  from points, compare models, and save predictions and a reusable classifier.

From the repository root, install the notebook dependencies and launch JupyterLab:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Open `docs/examples/` in JupyterLab. Add the `xgboost` extra to run that optional
classifier too: `python -m pip install -e ".[notebooks,xgboost]"`.
See the [notebook setup guide](docs/examples/index.md) for details.

## Documentation

Published documentation is available at `https://obia.sefa.ai`.

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
