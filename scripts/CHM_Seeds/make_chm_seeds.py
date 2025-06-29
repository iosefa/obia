#!/usr/bin/env python3
"""
make_chm_seeds.py
────────────────────────────────────────────────────────────
Extract seed points from a canopy-height model (CHM) by
detecting local maxima above a height threshold.

Outputs
-------
• chm_seeds.gpkg  – point layer with columns:
    - id          (integer)
    - ch_max      (height value at the peak, same units as CHM)

Peaks are detected in raster (row, col) space, then converted
to map coordinates via the CHM affine transform.
"""

# ───────────────────────────── USER PATHS ─────────────────────────
CHM_RASTER   = "/Users/iosefa/repos/obia/docs/example_data/site_2/chm.tif"                 # input CHM raster
SEEDS_GPKG   = "/Users/iosefa/repos/obia/scripts/CHM_Seeds/chm_seeds.gpkg"          # output file
LAYER_NAME   = "chm_seeds"                        # layer inside GPKG
# ────────────────────────── PARAMETERS ────────────────────────────
H_MIN_M      =  2.5     # ignore peaks < this height (units = CHM units)
MIN_DIST_PX  =  3       # enforce ≥ this many pixels between peaks
GAUSS_SIGMA  =  1       # Gaussian blur σ before peak finding (0 ⇒ none)
# ------------------------------------------------------------------

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import dataset_features
from rasterio.transform import xy
from scipy.ndimage import gaussian_filter, maximum_filter

def detect_peaks(arr: np.ndarray,
                 h_min: float,
                 min_dist_px: int,
                 sigma: float = 0) -> np.ndarray:
    """Return (row, col) indices of local maxima in *arr*."""
    if sigma > 0:
        arr = gaussian_filter(arr, sigma=sigma)

    # candidate peaks = cells that equal the local maximum
    local_max = (arr == maximum_filter(arr, size=2 * min_dist_px + 1))
    # apply height threshold
    peaks = np.logical_and(local_max, arr >= h_min)

    return np.column_stack(np.where(peaks))

def main() -> None:
    chm_path = Path(CHM_RASTER)
    if not chm_path.exists():
        raise SystemExit(f"✗ CHM raster not found: {chm_path}")

    print("• reading CHM raster …")
    with rasterio.open(chm_path) as src:
        chm = src.read(1, masked=True).filled(np.nan)
        transform = src.transform
        crs = src.crs

    print("• detecting peaks …")
    peak_rc = detect_peaks(chm, H_MIN_M, MIN_DIST_PX, GAUSS_SIGMA)
    if peak_rc.size == 0:
        raise SystemExit("No peaks found – adjust H_MIN_M or check CHM.")

    # map (row, col) → coordinate (x, y)
    rows, cols = peak_rc[:, 0], peak_rc[:, 1]
    xs, ys = xy(transform, rows, cols, offset="center")
    heights = chm[rows, cols]

    gdf = gpd.GeoDataFrame(
        {"id": np.arange(len(xs)), "ch_max": heights},
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs,
    )

    out_path = Path(SEEDS_GPKG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, layer=LAYER_NAME, driver="GPKG", overwrite=True)

    print(f"✓ wrote {len(gdf):,} CHM seed points → {out_path}")

if __name__ == "__main__":
    main()