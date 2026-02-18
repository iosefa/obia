#!/usr/bin/env python3
"""
make_density_seeds.py
────────────────────────────────────────────────────────────
Detect local maxima in a LiDAR-derived density raster and
export them as seed points (GeoPackage).

Typical use-case: create seed points where the CHM peaks
may be sparse but canopy-surface return density is high.

Outputs
-------
• den_seeds.gpkg – point layer with:
     id          (integer)
     den_max     (density value at the peak)

Author : <you>
Date   : 2025-MM-DD
"""

# ─────────────────────────── USER PATHS ────────────────────────────
DENSITY_RASTER = "/Users/iosefa/repos/obia/docs/example_data/full/density.tif"
SEEDS_GPKG     = "/Users/iosefa/repos/obia/scripts/final/den_seeds.gpkg"
LAYER_NAME     = "den_seeds"                 # name of layer inside GPKG

# ───────────────────────── PARAMETERS ──────────────────────────────
D_MIN        =  4.5      # ignore peaks with density < this value
MIN_DIST_PX  =  4        # enforce ≥ this many pixels between peaks
GAUSS_SIGMA  =  2        # Gaussian blur σ before peak finding (0 ⇒ none)
# -------------------------------------------------------------------

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from scipy.ndimage import gaussian_filter, maximum_filter


def detect_peaks(arr: np.ndarray,
                 v_min: float,
                 min_dist_px: int,
                 sigma: int = 0) -> np.ndarray:
    """Return (row, col) indices of local maxima in *arr* ≥ v_min."""
    if sigma > 0:
        arr = gaussian_filter(arr, sigma=sigma)

    local_max = (arr == maximum_filter(arr, size=2 * min_dist_px + 1))
    peaks = np.logical_and(local_max, arr >= v_min)
    return np.column_stack(np.where(peaks))


def main() -> None:
    raster_path = Path(DENSITY_RASTER)
    if not raster_path.exists():
        raise SystemExit(f"✗ density raster not found: {raster_path}")

    print("• reading density raster …")
    with rasterio.open(raster_path) as src:
        den = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
        crs = src.crs

    print("• detecting peaks …")
    peak_rc = detect_peaks(den, D_MIN, MIN_DIST_PX, GAUSS_SIGMA)
    if peak_rc.size == 0:
        raise SystemExit("No density peaks found — lower D_MIN or check raster.")

    rows, cols = peak_rc[:, 0], peak_rc[:, 1]
    xs, ys = xy(transform, rows, cols, offset="center")
    dvals = den[rows, cols]

    gdf = gpd.GeoDataFrame(
        {"id": np.arange(len(xs)), "den_max": dvals},
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs,
    )

    out_path = Path(SEEDS_GPKG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, layer=LAYER_NAME, driver="GPKG", overwrite=True)

    print(f"✓ wrote {len(gdf):,} density-seed points → {out_path}")


if __name__ == "__main__":
    main()