# Example notebooks

These notebooks use real imagery and land-cover reference data from the
[Chesapeake Land Cover dataset](https://lila.science/datasets/chesapeakelandcover/).
Each runs independently with the sample files included in OBIA; no data download
is needed after installation.

| Notebook | What it demonstrates |
| --- | --- |
| [1. Raster segmentation](01_segmentation.ipynb) | Compare segment boundaries, create object polygons, and export raster IDs and a GeoPackage. |
| [2. Raster object features](02_raster_features.ipynb) | Measure band distributions, a normalized difference, shape, texture, and pixel validity within an analysis area. |
| [3. Object classification](03_classification.ipynb) | Sample reference labels, compare classifiers with spatial validation, and evaluate and map a separate test area. |
| [4. Pixel classification](04_pixel_classification.ipynb) | Learn band values at labelled points, predict a class raster in another area, and save georeferenced results. |

## Run a notebook

Use Python 3.11 or newer and a [repository checkout](../installation.md#install-from-source).
Install from that checkout so the examples and API match. From its root:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Open a notebook under `docs/examples/`, select the Python kernel for that
environment, and choose **Run → Run All Cells**. If Jupyter is running
in a different environment, register this environment as a kernel first:

```bash
python -m ipykernel install --user --name obia --display-name "Python (OBIA)"
```

Select **Python (OBIA)** in Jupyter. The notebooks can run in any order;
none reads another notebook's outputs. Results go into
`example_outputs/<notebook-name>/` relative to the kernel's working directory.
Rerunning replaces those example files. XGBoost is included in the standard
installation and runs alongside the other classifiers.

## Included data

OBIA includes two 1024 × 1024 pixel crops from Maryland: one from an upstream
training tile and one from a separate test tile. Each pairs four-band NAIP 2013
imagery with a land-cover reference raster on the same 1 m grid in EPSG:26918.
The four GeoTIFFs total about 5.3 MiB. Cropping preserves the original pixel values
and band order; select imagery bands using indices `0`, `1`, `2`, and `3`.

```python
import rasterio

from obia.datasets import chesapeake_sample
from obia.handlers.geotif import open_geotiff

sample = chesapeake_sample("train")  # Use "test" for the other area.
image = open_geotiff(sample["image"])
with rasterio.open(sample["reference"]) as src:
    reference = src.read(1)
print(sample["class_names"])
print(sample["metadata"]["tile_id"])
image.rasterio_obj.close()  # Pixel arrays and spatial metadata remain available.
```

`chesapeake_sample()` returns local file paths, class names, and provenance.
The files also ship in the Python package. Their `metadata.json` records source
URLs, crop windows, CRS, transforms, and file and pixel checksums. The repository's
`scripts/prepare_chesapeake_sample.py` can reproduce them from the source rasters.

| Reference code | Class |
| --- | --- |
| 1 | Water |
| 2 | Tree canopy / forest |
| 3 | Low vegetation / field |
| 4 | Barren land |
| 5 | Impervious (other) |
| 6 | Impervious (road) |
| 15 | NoData |

The reference is a published land-cover map. The classification notebook samples
points from it and measures agreement on objects with consistent sampled labels.
These points are not independent field observations. Mixed objects are excluded
from scoring, some classes have few examples, and these crops are not a
representative accuracy sample or the full benchmark. See the
[validation guide](../usage/validation.md) for study design and map accuracy assessment.

### Sources and attribution

Imagery: USDA NAIP. Reference labels: Chesapeake Conservancy. Dataset preparation:
Robinson et al. (2019), *Large Scale High-Resolution Land Cover Mapping with
Multi-Resolution Data*, CVPR, [doi:10.1109/CVPR.2019.01301](https://doi.org/10.1109/CVPR.2019.01301).

The publisher identifies NAIP imagery as public domain and requires source credit
for reference-label reuse. Its notice also describes limitations and disclaims
warranties. The bundled `obia/datasets/chesapeake/README.md` records attribution;
consult the [publisher's data terms](https://lila.science/datasets/chesapeakelandcover/)
when reusing the files. OBIA's MIT code license does not replace those terms.

## Use your own data

Replace the sample path with your GeoTIFF, choose numeric band indices, and adjust
segmentation parameters for its resolution and object sizes. The feature notebook
also accepts existing polygons. For classification, replace reference-map points
with your labelled points and choose training, validation, and test areas for
your prediction task.

The loader's optional file-band selection is one-based. Subsequent OBIA band
selections are zero-based positions in the loaded image.

## Check notebook changes

Executed outputs let readers see plots on GitHub and in these docs. CI reruns
every notebook in a fresh kernel. To check execution without replacing the
checked-in outputs:

```bash
mkdir -p /tmp/obia-notebook-checks
jupyter nbconvert --to notebook --execute docs/examples/*.ipynb \
  --output-dir /tmp/obia-notebook-checks --ExecutePreprocessor.timeout=180
```

Review the plots and save a clean top-to-bottom execution after changes. Generated
rasters, GeoPackages, and models belong in `example_outputs/`, which is ignored by
Git and the documentation build.
