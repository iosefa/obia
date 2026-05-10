# Installation

OBIA is installed with `pip`. The core package includes the raster, vector, segmentation, and classification dependencies used by the standard OBIA workflow.

## Requirements

OBIA requires Python 3.11 or newer.

Runtime dependencies are installed with the package, including:

- `rasterio`
- `geopandas`
- `numpy`
- `scikit-image`
- `scikit-learn`

For most users, installing from PyPI is the right starting point.

## Install From PyPI

```bash
pip install obia
```

Verify the install:

```bash
python -c "import obia; print(obia.__name__)"
```

## Install With Point-Cloud Support

Point-cloud workflows need PDAL. Install PDAL in a conda environment first, then install OBIA with `pip` inside that environment:

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

## Optional Extras

Install documentation dependencies with:

```bash
pip install "obia[docs]"
```

Install SHAP explanation dependencies with:

```bash
pip install "obia[explain]"
```

Install experimental object-detection dependencies with:

```bash
pip install "obia[detection]"
```

## Developer Install

For local development, clone the repository and create the conda environment:

```bash
git clone https://github.com/iosefa/obia.git
cd obia
conda env create -f environment.yml
conda activate obia
```

The environment installs OBIA in editable mode and includes the documentation, point-cloud, detection, explanation, and test dependencies used during development.

If you already have a compatible environment and only want the editable package install:

```bash
pip install -e .
```

To include documentation and test dependencies in an existing environment:

```bash
pip install -e ".[docs,test]"
```

## Preview The Documentation

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
