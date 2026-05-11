# Segmentation

This page covers the standard OBIA segmentation path: read a raster, create image objects, calculate object features, and write the resulting GeoDataFrame.

## Load a Raster

```python
from obia.handlers.geotif import open_geotiff

image = open_geotiff("/path/to/image.tif")
```

`open_geotiff()` reads all bands by default. To read only selected GeoTIFF bands, pass one-based Rasterio band numbers:

```python
image = open_geotiff("/path/to/image.tif", bands=[1, 2, 3, 4])
```

Once loaded, OBIA uses zero-based band indexes for `segmentation_bands` and `statistics_bands`.

## Run SLIC Segmentation

```python
from obia.segmentation.segment import segment

segments = segment(
    image,
    segmentation_bands=[0, 1, 2],
    statistics_bands=[0, 1, 2, 3],
    method="slic",
    n_segments=3000,
    compactness=10,
)
```

The returned object stores:

- `segments._segments`: the raw label image
- `segments.segments`: a GeoDataFrame of segment polygons and features
- `segments.method`: the segmentation method
- `segments.params`: segmentation parameters passed to the algorithm

## Run Quickshift Segmentation

```python
segments = segment(
    image,
    segmentation_bands=[0, 1, 2],
    method="quickshift",
    kernel_size=3,
    max_dist=6,
    ratio=0.5,
)
```

Use SLIC as the default starting point. Quickshift can be useful for texture-rich imagery, but it is generally more sensitive to parameters.

## Choose Bands

`segmentation_bands` controls which bands define object boundaries. Use bands with strong spatial structure and class-relevant contrast.

`statistics_bands` controls which bands are summarized after the segments are created. These can include bands that were not used for segmentation.

For example, segment on RGB but calculate features from RGB plus a height or vegetation index band:

```python
segments = segment(
    image,
    segmentation_bands=[0, 1, 2],
    statistics_bands=[0, 1, 2, 3],
    method="slic",
    n_segments=3000,
    compactness=10,
)
```

## Save Segment Objects

```python
segments.segments.to_file("segments.gpkg", driver="GPKG")
```

GeoPackage is a practical default because it preserves geometry, CRS, and tabular feature columns in one file.

## Add Other Feature Sources

The segment GeoDataFrame is the common feature table. After segmentation, additional feature sources can be joined to the same rows. For point clouds:

```python
from obia.pointcloud import add_pointcloud_features

segments_gdf = add_pointcloud_features(
    segments.segments,
    pointcloud="/path/to/points.laz",
    metrics=["height", "intensity", "density"],
)
```

Use the enriched `segments_gdf` for labelling and classification.

## Inspect Results

A segmentation run should be judged spatially before classification. Check whether object boundaries are meaningful for the classes you want to predict. If a segment regularly crosses class boundaries, tune segmentation before training a classifier.

The most important SLIC parameters are usually:

- `n_segments`: approximate target number of segments
- `compactness`: balance between color similarity and spatial regularity
- `mask`: optional mask to exclude pixels from segmentation

Increase `n_segments` for smaller objects. Increase `compactness` for more regular shapes.
