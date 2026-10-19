# Raster Object Features

Features describe each object's raster values, shape, and texture. Add them to
segment polygons with `extract_features()`. You can change the measurements
without repeating segmentation; the result preserves the original geometry and
attributes and adds numeric feature columns.

All feature inputs use **zero-based integer band indices into the loaded image**.
If a file was loaded with `open_geotiff(path, bands=[4, 1, 2])`, feature bands
`0`, `1`, and `2` refer to those loaded bands in that order. The loader's file-band
argument is one-based. Choose band positions explicitly for your raster.

## Extract object features

This example uses a raster with at least four loaded bands and a projected CRS.
Choose the band positions for your image. For a geographic CRS, use
`shape_units="pixel"` to measure shape relative to the raster grid.

```python
from obia.features import extract_features
from obia.handlers.geotif import open_geotiff
from obia.segmentation import create_segments

image = open_geotiff("image.tif")
polygons = create_segments(
    image, segmentation_bands=[0, 1, 2], method="slic", n_segments=1000,
    convert2lab=False,
)

objects = extract_features(
    polygons,
    image,
    bands=[0, 1, 2, 3],
    spectral=("mean", "std", "median"),
    percentiles=(10, 90),
    indices={"normalized_difference": (3, 0)},
    shape=("area", "circularity"),
)

objects.to_file("objects.gpkg", driver="GPKG")
```

Feature column names record the calculation and band positions. Depending on
the measurements you request, examples include:

- `b0_median`, `b3_p90`, `b0_valid_fraction`
- `index_ndvi_b3_b0_mean`, `index_vari_b0_b1_b2_std`
- `shape_area`, `shape_elongation`
- `b0_glcm_entropy_d1`, `b0_glcm_entropy_d4`
- `b0_local_entropy_r3_mean`, `b0_local_entropy_r3_std`

Calling `extract_features()` again replaces requested columns with freshly
calculated values and preserves unrelated columns. It does not modify the input
table or raster. `spectral=(), percentiles=()` disables distributions;
`bands=()` calculates distributions only for requested indices. Shape and texture
are opt-in. Without options, extraction returns all spectral statistics below
and the 10th and 90th percentiles for all loaded bands.

## Band statistics

| Feature | Definition |
| --- | --- |
| `mean` | Arithmetic mean of finite valid pixels |
| `std`, `variance` | Population standard deviation and variance (`ddof=0`) |
| `min`, `max` | Minimum and maximum valid values |
| `median` | Median valid value |
| `iqr` | 75th percentile minus 25th percentile |
| `mad` | Median absolute deviation from the median, without a scaling factor |
| `skewness` | Standardized third moment, using SciPy's `bias=True` convention |
| `kurtosis` | Excess kurtosis (normal distribution: 0), `fisher=True, bias=True` |
| `valid_count` | Number of valid pixels within the polygon |
| `valid_fraction` | Valid count divided by polygon-covered pixel centers within the raster |

`percentiles=(5, 50, 95)` adds `p5`, `p50`, and `p95` columns using linear
interpolation. Fractional requests are supported: `2.5` produces `p2p5`.

Constant samples have zero standard deviation, variance, IQR, and MAD. Their
skewness and kurtosis are undefined and returned as NaN. Objects with no valid
samples have count 0 and NaN distribution measurements. Their valid fraction is
0 when they cover raster pixels, and NaN when they cover none.

## Spectral indices and color

`indices` maps each calculation name to an ordered tuple of band indices. For
example, `{"ndvi": (3, 0)}` means `(b3 - b0) / (b3 + b0)`. The caller chooses the
correct positions; specifying an index name never discovers or reorders bands.

In the formulas below, `a`, `b`, and `c` mean pixel values from the first, second,
and third **numeric band positions supplied in the tuple**, respectively. Wavelength descriptions explain
the conventional index definitions; they are not API parameter names.

| Name | Formula | Required input order |
| --- | --- | --- |
| `normalized_difference` | `(a-b)/(a+b)` | Any two chosen bands |
| `ndvi` | `(a-b)/(a+b)` | NIR, red |
| `gndvi` | `(a-b)/(a+b)` | NIR, green |
| `ndre` | `(a-b)/(a+b)` | NIR, red edge |
| `ndwi` | `(a-b)/(a+b)` | Green, NIR (McFeeters water index) |
| `mndwi` | `(a-b)/(a+b)` | Green, SWIR |
| `ndmi` | `(a-b)/(a+b)` | NIR, SWIR1 |
| `nbr` | `(a-b)/(a+b)` | NIR, SWIR2 |
| `ndbi` | `(a-b)/(a+b)` | SWIR1, NIR |
| `ratio` | `a/b` | Any two chosen bands |
| `difference` | `a-b` | Any two chosen bands |
| `exg` | `(2*b-a-c)/(a+b+c)` | Red, green, blue; **normalized** excess green |
| `vari` | `(b-a)/(b+a-c)` | Red, green, blue |
| `brightness` | Arithmetic mean | One or more chosen bands |

Each index is computed **per pixel**, then summarized using the requested
spectral statistics and percentiles. Mean pixel NDVI is generally different from
NDVI computed from object band means. Inputs retain their original values;
segmentation normalization is never used for feature calculations. Apply any
required radiometric calibration before extracting features.

Pixels need valid finite values in every required band. Zero denominators and
nonfinite results become NaN, which are excluded from summaries. Valid results
are not clipped to an assumed index range. Source arrays are unchanged and
integer inputs are converted before arithmetic to avoid unsigned subtraction.

To obtain an index image directly:

```python
from obia.features import compute_index

index_image = compute_index(image.img_data, "ndvi", bands=(3, 0))
```

`compute_index()` honors masked arrays and nonfinite values. When passing a plain
array directly, apply any file NoData mask first. `extract_features()` applies
the `Image` source validity masks automatically.

Index definitions can be compared with the original references collected in
[Awesome Spectral Indices](https://www.nature.com/articles/s41597-023-02096-0).

## Size and shape

Pass `shape=True` for all measurements, or a list such as
`shape=("area", "circularity", "solidity")`.

| Feature | Definition |
| --- | --- |
| `area` | Polygon area, excluding holes |
| `perimeter` | All boundary lengths, including holes |
| `major_axis_length`, `minor_axis_length` | Long and short sides of the minimum-area oriented bounding rectangle |
| `elongation` | Major axis length / minor axis length |
| `circularity` | `4 * pi * area / perimeter**2` |
| `solidity` | Area / convex-hull area |
| `rectangularity` | Area / oriented bounding rectangle area |
| `orientation` | Major rectangle axis angle in degrees, in `[0, 180)` |
| `hole_count` | Number of interior polygon rings, summed for multipolygons |

`shape_units="map"` requires a projected CRS. Area is in squared CRS units,
lengths in CRS units, and angles run counterclockwise from the positive x-axis.
Units may be feet rather than meters; inspect the raster CRS.
`shape_units="pixel"` first transforms geometry through the full inverse affine,
including rotation, shear, and unequal pixel sizes. Lengths are in pixels, area
in square pixels, and angles run from increasing columns toward increasing rows.
For tied rectangle axes, the smaller axis angle is chosen.

Measurements use the supplied polygon geometry. Feature masks affect sampled
pixels but do not redefine its shape. Empty geometries yield NaN shape features.
These formulas are explicit because software packages use different meanings for
names such as compactness and roundness. Compare [ENVI's shape definitions](https://www.nv5geospatialsoftware.com/docs/AttributeList.html).

## Texture

Texture describes spatial variation within an object. A gray-level co-occurrence
matrix (GLCM) counts pairs of pixel values at specified distances and directions.
Local entropy measures variation within a neighborhood.

Add texture measurements to an object table with `TextureConfig`. All distances
and radii are in **pixels**:

```python
from obia.features import TextureConfig

objects = extract_features(
    objects, image,
    spectral=(), percentiles=(),  # Keep the measurements already in the table.
    texture=TextureConfig(
        bands=(0, 3),
        features=("contrast", "homogeneity", "entropy"),
        distances=(1, 2, 4),
        levels=32,
        entropy_radii=(1, 3),
    ),
)
```

GLCM pairs count only when **both endpoints** are finite valid pixels inside the
same object. Polygon holes, other objects, NoData, and pixels outside a supplied
mask never contribute artificial zero-valued observations. Pair distances describe
endpoint separation; intermediate pixels do not determine pair validity.

Values are quantized into `levels` uniform bins, from 2 to 256. By default, a
separate range is fitted to each band's finite included pixels over the **whole
image**, then reused for every object. It is not fitted separately to each object.
Use `value_range=(0, 10000)` to fix the bounds explicitly, especially when comparing
scenes. Out-of-range values are clipped before binning. Constant bands occupy
one bin. Integer rasters are not cast directly to 8-bit values.

The available `features` are:

- `contrast`, `dissimilarity`, `homogeneity`, `ASM`, `energy`, `correlation`, using
  [scikit-image's definitions](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops).
- `entropy`: `-sum(p * log2(p))` over nonzero GLCM probabilities.
- `mean`: `sum(i * p[i,j])`, the first marginal mean.
- `variance`: `sum((i - mean_i)**2 * p[i,j])`.
- `maximum_probability`: the largest GLCM probability.
- `cluster_shade`, `cluster_prominence`: third and fourth moments of
  `i + j - mean_i - mean_j`, respectively.

Matrices are normalized, and `symmetric=True` includes both pair orders. Each
distance is reported separately. The default `angle_aggregation="mean"` averages
feature values over directions that have valid pairs, giving each such direction
equal weight. `angle_aggregation="none"` produces `_a0`, `_a1`, etc., referring to
positions in the supplied `angles` tuple. Angles are in radians, measured from
increasing columns toward increasing rows.

A direction with no valid pairs produces NaN. A distance with no usable direction
also produces NaN. Constant valid pairs have contrast 0 and correlation 1,
following scikit-image's convention.

`entropy_radii=(1, 3)` additionally calculates local Shannon entropy in bits using
disk footprints of those radii. Only valid pixels within the object participate
in each window. The mean and population standard deviation across valid window
centers become object features. Set `features=()` for local entropy alone.

Configuration and fitted bounds are available in `objects.attrs["obia_features"]`.
This is in-memory metadata; save it separately if needed alongside a GeoPackage.

## Extract features during segmentation

`segment()` combines boundary creation and feature extraction. Pass
`feature_options` to choose measurements in the same call:

```python
from obia.segmentation.segment import segment

result = segment(
    image, segmentation_bands=[0, 1, 2], statistics_bands=[0, 1, 2, 3],
    method="slic", n_segments=1000, convert2lab=False,
    feature_options={
        "spectral_features": ("mean", "std", "median", "valid_count"),
        "percentiles": (10, 90),
        "indices": {"ndvi": (3, 0)},
        "shape_features": True,
        "texture_config": TextureConfig(bands=(0,), distances=(1, 2)),
    },
)
objects = result.segments
```

These options are forwarded to `create_objects()`. They take precedence over
individual `calc_*` switches. Without `feature_options`, `segment()` uses its
default statistics and texture measurements, with names such as `b0_mean` and
`b0_contrast`.

For `create_objects()`, `calculate_spectral=False` disables band and index
distributions, and `calculate_textural=False` disables texture. This permits
shape-only extraction. `texture_config` selects texture measurements;
`spectral_features` selects spectral statistics. Without an explicit
texture band list, `create_objects()` uses `textural_bands` or all loaded bands;
the standalone `extract_features()` instead defaults texture to its `bands` selection.

## Missing data and spatial alignment

Features are calculated from `image.img_data`, so they also work with in-memory
images and after the source file has been closed. `open_geotiff()` retains source
NoData/dataset masks in `image.valid_mask`, in loaded band order. A manually
constructed `Image` can supply the same optional boolean array, with shape
`(height, width, bands)`. Nonfinite values and NumPy masked-array samples are
always excluded.

An additional `mask` has shape `(height, width)` and applies to all bands. Polygon
sampling uses pixel centers and honors holes. Segments must have the same CRS as
the raster; reproject them explicitly before extraction. Polygons outside the
raster retain their geometry and receive undefined raster measurements.

Pass the result to the [labelling and classification workflow](classification.md).
Inspect missing values before training; tiny objects may have no texture pairs,
and some indices are undefined where their denominators vanish.
