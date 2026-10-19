# Raster loading API

`open_geotiff()` returns an instance of `Image`, a class defined by OBIA.
It keeps the pixel array together with its coordinate reference system,
transform, source dataset, and valid-pixel mask. `Image.to_image()` creates
a Pillow image for display.

## Raster data

::: obia.handlers.geotif

## Bundled example data

::: obia.datasets.chesapeake_sample
