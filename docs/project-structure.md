# Project Structure

## Package modules

- `obia/handlers`: raster read/write helpers and `Image` wrapper
- `obia/segmentation`: segmentation and segment statistics/object creation
- `obia/classification`: tabular ML classification on segment features
- `obia/detection`: RetinaNet-based detection dataset/model/train/predict code
- `obia/utils`: utility helpers for tiling, image transforms, cost surfaces, seed creation

## Repository areas

- `docs/`: MkDocs source files
- `docs/examples/`: maintained notebooks used by the documentation
- `tests/`: test files
- `pyproject.toml`: packaging metadata

## Cleanup note

Legacy experimental material was previously under `scripts/`, `notebooks/`, and the repo root. It has been removed from the maintained package surface.
