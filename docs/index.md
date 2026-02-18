# obia

Object-based image analysis tools for geospatial raster workflows.

## What this project is

`obia` is a Python package for:

- loading georeferenced rasters (`obia.handlers`)
- creating segment polygons and segment-level features (`obia.segmentation`)
- training/inference for detection models (`obia.detection`)
- segment classification with scikit-learn (`obia.classification`)
- utility workflows for tiling, seed generation, and cost-surface construction (`obia.utils`)

## Current scope

This repository is in active cleanup. The docs now describe the modules that exist in this repo today and avoid inherited `pyforestscan` content.

Use these pages first:

- Installation: `installation.md`
- Quickstart: `quickstart.md`
- Workflows: `workflows.md`
- API reference: `api/index.md`
