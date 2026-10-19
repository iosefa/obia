# Example Notebooks

These three notebooks demonstrate the raster workflow with small, reproducible
examples. Each creates a synthetic four-band GeoTIFF, runs independently, and
includes plots and saved GIS outputs. No external datasets or downloads are
needed after installation.

| Notebook | What it demonstrates |
| --- | --- |
| [1. Raster segmentation](01_segmentation.ipynb) | Compare SLIC, Felzenszwalb, and watershed; convert labels to object polygons; export segment IDs and a GeoPackage. |
| [2. Raster object features](02_raster_features.ipynb) | Add spectral summaries, a normalized-difference index, shape, texture, and mask-based validity measurements to existing polygons. |
| [3. Object classification](03_classification.ipynb) | Transfer point labels to objects, inspect spatial blocks and buffer exclusions, compare classifiers on the same split, and save a class map and fitted model. |

## Run Locally

Use Python 3.11 or newer and a checkout containing these notebooks. Install the
library and notebook dependencies from that checkout so the examples and API
match; an older PyPI release may not contain the features shown here.

From the repository root, in your Python environment:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Open a notebook under `docs/examples/`, select the Python kernel for that
environment, and choose **Run → Run All Cells**. If Jupyter is already running
in a different environment, register this environment as a kernel first:

```bash
python -m ipykernel install --user --name obia --display-name "Python (OBIA)"
```

Then select **Python (OBIA)** in Jupyter. The notebooks can be run in any order;
none reads another notebook's outputs. Files are saved under
`example_outputs/<notebook-name>/` relative to the kernel's working directory
(normally beside the notebook). Rerunning replaces those example files.

The classification notebook uses random forest, logistic regression, histogram
gradient boosting, and a custom SVM pipeline. To include XGBoost as well:

```bash
python -m pip install -e ".[notebooks,xgboost]"
```

Restart the kernel after changing dependencies. Without XGBoost, that comparison
is skipped and the rest of the notebook still runs.

## Use Your Own Data

Replace the synthetic-raster cell with `open_geotiff("your_image.tif")`, choose
numeric band indices for your raster, and adjust segmentation parameters for its
resolution and object sizes. The feature notebook also accepts existing polygon
data. The classification notebook shows where to substitute labelled points with
a `class` column, reprojected to the object CRS.

The loader's optional file-band selection is one-based. All subsequent OBIA band
selections are zero-based positions in the loaded image. Synthetic class scores
are demonstrations, not estimates of accuracy on real imagery. Choose sampling
units and separation distances for your prediction task; the
[validation guide](../usage/validation.md) covers grouped, spatial, and explicit
splits, model selection, and map accuracy assessment, with literature references.

## Check Notebook Changes

The repository keeps executed outputs so readers can see the plots on GitHub and
in the documentation. CI executes the notebooks again in fresh kernels. To run
the same check locally without replacing the checked-in notebooks:

```bash
mkdir -p /tmp/obia-notebook-checks
jupyter nbconvert --to notebook --execute docs/examples/*.ipynb \
  --output-dir /tmp/obia-notebook-checks --ExecutePreprocessor.timeout=180
```

Review plots and outputs after editing a notebook, and save a clean top-to-bottom
execution. Generated rasters, GeoPackages, and models belong in `example_outputs/`
and are ignored by Git and the documentation build.
