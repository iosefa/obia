# OBIA

[![PyPI](https://img.shields.io/pypi/v/obia.svg?release=0.1.4)](https://pypi.org/project/obia/)
[![PyPI Downloads](https://static.pepy.tech/badge/obia)](https://pepy.tech/projects/obia)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/obia?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/obia)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/obia/tests.yml?branch=main&label=tests)](https://github.com/iosefa/obia/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/iosefa/obia/docs.yml?branch=main&label=docs)](https://github.com/iosefa/obia/actions/workflows/docs.yml)
[![Contributors](https://img.shields.io/github/contributors/iosefa/obia.svg?label=contributors)](https://github.com/iosefa/obia/graphs/contributors)
[![License](https://img.shields.io/github/license/iosefa/obia)](https://github.com/iosefa/obia/blob/main/LICENSE)

Object-based image analysis tools for geospatial rasters.

OBIA groups raster pixels into objects, measures their properties, and classifies
them using labelled examples. Results are stored in a GeoDataFrame: a table of
polygons and their attributes that you can inspect, save, and use in GIS.

With OBIA, you can:

- Load GeoTIFFs with their coordinate reference system and spatial transform.
- Segment imagery into object polygons.
- Measure each object's band statistics, spectral indices, shape, and texture.
- Transfer point labels to objects and train a classifier.
- Evaluate models with grouped or spatial splits, buffers, and cross-validation.
- Add point-cloud measurements or process large rasters in tiles when needed.

## Install

Use Python 3.11 or newer. Install the published release with:

```bash
python -m pip install obia
```

The documentation follows `main`. For a matching source installation, optional
dependencies, or Docker setup, see the
[installation guide](https://obia.sefa.ai/installation/).

## Segment a raster

This example assumes at least four loaded bands. Adjust the numeric band indices
to match your image; `0` refers to the first loaded band.

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

segments = objects.segments
segments.to_file("segments.gpkg", driver="GPKG")
```

Each row of `segments` contains a polygon, a `segment_id`, and feature columns.
See the [segmentation guide](https://obia.sefa.ai/usage/segmentation/) for other
algorithms, masks, and parameter examples.

## Measure objects

Calculate additional features on existing polygons without repeating segmentation.
All bands use zero-based numeric positions in the loaded image. This example
uses a projected CRS for shape measurements; use `shape_units="pixel"` if your
raster has a geographic CRS:

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

## Label and classify objects

Provide labelled points with a `class` column, using the same coordinate system
as the segments:

```python
import geopandas as gpd

from obia.classification import classify
from obia.utils.utils import label_segments

labelled_points = gpd.read_file("/path/to/labelled_points.gpkg").to_crs(segments.crs)
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

For validation in new sites or areas, create and inspect a partition before
fitting. For example, with a `site_id` column identifying independent sites:

```python
from obia.classification import split_objects

partition = split_objects(training_segments, method="grouped", groups="site_id",
                          test_size=0.2, random_state=42)
print(partition.class_counts)
result = classify(segments, training_segments, method="rf", split=partition,
                  compute_reports=True, random_state=42)
```

Spatial blocks, geometry buffers, explicit train/test sets, and reusable
cross-validation folds are also supported. See the
[validation guide](https://obia.sefa.ai/usage/validation/) for examples and the
literature behind choosing a design that matches the prediction task.

Use a method name or supply a compatible estimator or pipeline. The
[classification guide](https://obia.sefa.ai/usage/classification/) covers model
choices, preprocessing, predictions, and evaluation.

## Example notebooks

The notebooks run independently using the bundled Chesapeake imagery and
reference maps, with plots and GIS exports:

- [Raster segmentation](docs/examples/01_segmentation.ipynb): compare algorithms,
  create object polygons, and export segment IDs and a GeoPackage.
- [Raster object features](docs/examples/02_raster_features.ipynb): spectral
  summaries, indices, shape, texture, and analysis masks.
- [Object classification](docs/examples/03_classification.ipynb): label objects
  from points, compare models, and save predictions and a reusable classifier.

From the repository root, install the notebook dependencies and launch JupyterLab:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Open `docs/examples/` in JupyterLab.
See the [notebook setup guide](docs/examples/index.md) for details.

## Documentation

Read the [documentation](https://obia.sefa.ai) for the full workflow. Optional
guides cover [point-cloud features](https://obia.sefa.ai/usage/pointcloud/),
[large rasters](https://obia.sefa.ai/usage/large-rasters/), and
[experimental detection](https://obia.sefa.ai/usage/detection/).

To preview the docs from a repository checkout:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Open `http://127.0.0.1:8000`. If the port is already in use:

```bash
mkdocs serve -a 127.0.0.1:8001
```

Documentation source lives in `docs/`.

## License

MIT (`LICENSE`)
