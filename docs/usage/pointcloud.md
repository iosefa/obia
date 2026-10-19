# Point Clouds

Point-cloud measurements describe height, intensity, and point density within
object polygons. Add them to the same table as your raster features, then use
the combined measurements for classification.

The recommended order is:

1. create segment objects from a raster
2. add point-cloud features to those segment objects
3. label training segments
4. classify the enriched segment table

## Install file readers

For LAS/LAZ files, install PDAL in a conda environment, then install OBIA with
`pip` in that environment. In-memory arrays and tables do not need file readers.

```bash
conda create -n obia-pointcloud -c conda-forge python=3.11 pdal python-pdal
conda activate obia-pointcloud
pip install obia
```

If you already have an environment, install PDAL from conda-forge before installing OBIA:

```bash
conda install -c conda-forge pdal python-pdal
pip install obia
```

The conda-forge PDAL package is usually more reliable than installing PDAL through pip because the Python bindings need the PDAL base library.

To include the optional `pyforestscan` reader, install the point-cloud extra after activating the same environment:

```bash
pip install "obia[pointcloud]"
```

## Add features to objects

```python
import geopandas as gpd

from obia.pointcloud import add_pointcloud_features

segments = gpd.read_file("segments.gpkg")

segments = add_pointcloud_features(
    segments,
    pointcloud="/path/to/points.laz",
    metrics=["height", "intensity", "density"],
)

segments.to_file("segments_with_pointcloud_features.gpkg", driver="GPKG")
```

The output preserves segment rows and appends `pc_*` feature columns. Segment geometries are used to select points; no new object boundaries are created.

If the point cloud has no CRS, pass one explicitly:

```python
segments = add_pointcloud_features(
    segments,
    pointcloud="/path/to/points.laz",
    crs=segments.crs,
)
```

## Use in-memory points

Point clouds can also be passed as a DataFrame:

```python
import pandas as pd

points = pd.DataFrame(
    {
        "x": [0.25, 0.75, 1.25],
        "y": [0.25, 0.75, 0.25],
        "z": [2.0, 4.0, 10.0],
        "intensity": [5.0, 7.0, 9.0],
    }
)

segments = add_pointcloud_features(
    segments,
    points,
    crs=segments.crs,
)
```

GeoDataFrames and NumPy arrays are also supported.

Structured arrays can use `X`, `Y`, `Z`, and `Intensity` fields. Plain NumPy arrays are interpreted as columns in this order:

```text
x, y, z, intensity
```

## Feature columns

Height metrics use the `pc_z_*` prefix:

- `pc_z_min`
- `pc_z_max`
- `pc_z_mean`
- `pc_z_median`
- `pc_z_std`
- `pc_z_p5`, `pc_z_p25`, `pc_z_p50`, `pc_z_p75`, `pc_z_p95`

Intensity metrics use the `pc_intensity_*` prefix. Density features include:

- `pc_point_count`
- `pc_density`

## Use the features in classification

Label the object table after adding the measurements:

```python
from obia.utils.utils import label_segments

training_segments, mixed_segments = label_segments(
    segments,
    labelled_points,
)
```

Then train a classifier:

```python
from obia.classification import classify

result = classify(
    segments,
    training_segments,
    method="rf",
    n_estimators=300,
)
```

The predicted output keeps the original segment geometry and the added point-cloud columns.
