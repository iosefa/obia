# Installation

OBIA requires Python 3.11 or newer. The core installation includes the
dependencies for loading rasters, creating objects, extracting features, and
training classifiers, including XGBoost.

## Install a release

```bash
python -m pip install obia
```

Verify the install:

```bash
python -c "import obia; print(obia.__name__)"
```

## Install from source

This documentation follows the repository's `main` branch. To use the same code
as the guides and notebooks, install from a checkout:

```bash
git clone https://github.com/iosefa/obia.git
cd obia
python -m pip install -e .
```

The `-e` option makes this an editable installation: changes in the checkout are
available without reinstalling the package.

## Run the notebooks

From the repository root, install the notebook dependencies and launch JupyterLab:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Open a notebook under `docs/examples/`. See the
[notebook guide](examples/index.md) for kernel setup and the available workflows.

## Optional dependencies

Install an extra for the workflow you need. In an editable source installation,
replace `obia[extra]` with `-e ".[extra]"` in these commands.

| Extra | Install command | Purpose |
| --- | --- | --- |
| SHAP | `python -m pip install "obia[explain]"` | Feature-attribution explanations |
| Documentation | `python -m pip install "obia[docs]"` | Build and preview these docs |
| Detection | `python -m pip install "obia[detection]"` | Experimental bounding-box detection |

SHAP requires version 0.51 or newer. Point-cloud file readers also need system
libraries, so follow the setup below if you plan to use them.

## Point-cloud support

For LAS/LAZ files, install PDAL in a conda environment, then install OBIA with
`pip` in that environment. In-memory point arrays and tables do not need these
file-reading dependencies.

```bash
conda create -n obia-pointcloud -c conda-forge python=3.11 pdal python-pdal
conda activate obia-pointcloud
pip install obia
```

This gives OBIA access to the conda-forge PDAL Python bindings for LAS/LAZ reading.

If you also want the optional `pyforestscan` reader, install the point-cloud extra after activating the same environment:

```bash
pip install "obia[pointcloud]"
```

## Docker

Build the image from the repository root:

```bash
docker build -t obia .
```

Release images are published to Docker Hub as `iosefa/obia`:

```bash
docker pull iosefa/obia:latest
```

The Docker image includes OBIA, PDAL, the PDAL Python bindings, and `pyforestscan`.

Start a Python session in the downloaded image:

```bash
docker run --rm -it iosefa/obia:latest
```

Mount a local data directory when working with rasters, vectors, or point clouds:

```bash
docker run --rm -it \
  -v "$PWD:/data" \
  iosefa/obia:latest
```

If you built the image locally with `docker build -t obia .`, use `obia` as the
image name instead.

## Development environment

For local development, clone the repository and create the conda environment:

```bash
git clone https://github.com/iosefa/obia.git
cd obia
conda env create -f environment.yml
conda activate obia
```

The environment installs OBIA in editable mode and includes the notebook, documentation, point-cloud, detection, explanation, XGBoost, and test dependencies used during development.

For a smaller development setup, install documentation and test dependencies
into your own Python environment from the repository root:

```bash
python -m pip install -e ".[docs,test]"
```

## Preview the documentation

Run the local documentation server from the repository root:

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000` in a browser. If that port is already in use, choose another port:

```bash
mkdocs serve -a 127.0.0.1:8001
```

To build the static site:

```bash
mkdocs build
```

The generated HTML is written to `site/`.
