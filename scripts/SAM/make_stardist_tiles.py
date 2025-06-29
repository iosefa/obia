#!/usr/bin/env python3
# make_stardist_tiles.py
#
# 1. For every SAM probability raster in tmp_prob/, build a 3‑band composite:
#       band‑1  = heavily‑smoothed SAM probability  (0‑1)
#       band‑2  = CHM normalised 0‑1 (per chip)
#       band‑3  = 1 – cost  (so inside crown ≈ 1, boundary ≈ 0)
# 2. Export the pixel with max smoothed probability as a point (crown centre).
#
# Requires: rasterio, numpy, scipy, geopandas, tqdm
# ---------------------------------------------------------------------------

from pathlib import Path
import re, os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# ────────────── CONFIG ─────────────────────────────────────────────────────
BASE_DIR        = Path("/Users/iosefa/repos/obia/scripts/SAM")
PROB_DIR        = BASE_DIR / "tmp_prob2"
COST_FILE       = BASE_DIR / "cost2.tif"
CHM_FILE        = BASE_DIR / "chm2.tif"

OUT_TILE_DIR    = BASE_DIR / "composite_tiles_ns2"
OUT_CENTRES_GPKG = BASE_DIR / "crown_centres2.gpkg"
SMOOTH_SIGMA    = 10           # pixels – “huge” smoothing
PCTL_LOW_HIGH   = (5, 95)     # for CHM / cost normalisation
# ---------------------------------------------------------------------------

OUT_TILE_DIR.mkdir(exist_ok=True)

# Regex to pull cluster id
_CLUSTER_RE = re.compile(r"seed_(\d+)", flags=re.IGNORECASE)

# Prepare centre‑point collector
centre_records = []

# Open static rasters once
cost_src = rasterio.open(COST_FILE)
chm_src  = rasterio.open(CHM_FILE)

for prob_path in tqdm(sorted(PROB_DIR.glob("seed_*.tif")), desc="tiles"):

    with rasterio.open(prob_path) as prob_src:
        cluster_id = int(_CLUSTER_RE.search(prob_path.name).group(1))

        # ---------------- band‑1: SMOOTHED SAM PROBABILITY -----------------
        prob = prob_src.read(1, masked=True).filled(np.nan).astype("float32")

        # nodata‑aware Gaussian
        nodata = np.isnan(prob)
        valid  = (~nodata).astype(float)
        prob_s = gaussian_filter(np.nan_to_num(prob), SMOOTH_SIGMA)
        valid_s = gaussian_filter(valid, SMOOTH_SIGMA)
        prob_s = np.where(valid_s == 0, np.nan, prob_s / valid_s)  # keep n/a
        prob_s = np.clip(prob_s, 0, 1)                             # safety

        # centre pixel (row, col) where prob_s is max
        centre_rc = np.unravel_index(np.nanargmax(prob_s), prob_s.shape)
        centre_xy = rasterio.transform.xy(prob_src.transform,
                                          centre_rc[0], centre_rc[1],
                                          offset='center')

        # ---------------- band‑2: CHM normalised 0‑1 ------------------------
        chm_chip = np.empty_like(prob_s, dtype="float32")
        reproject(
            rasterio.band(chm_src, 1), chm_chip,
            src_transform=chm_src.transform, src_crs=chm_src.crs,
            dst_transform=prob_src.transform, dst_crs=prob_src.crs,
            dst_nodata=np.nan, resampling=Resampling.bilinear
        )
        vmin, vmax = np.nanpercentile(chm_chip, PCTL_LOW_HIGH)
        chm_norm = np.clip((chm_chip - vmin) / (vmax - vmin + 1e-6), 0, 1)

        # ---------------- band‑3: 1 – COST normalised ----------------------
        cost_chip = np.empty_like(prob_s, dtype="float32")
        reproject(
            rasterio.band(cost_src, 1), cost_chip,
            src_transform=cost_src.transform, src_crs=cost_src.crs,
            dst_transform=prob_src.transform, dst_crs=prob_src.crs,
            dst_nodata=np.nan, resampling=Resampling.bilinear
        )
        vmin, vmax = np.nanpercentile(cost_chip, PCTL_LOW_HIGH)
        cost_norm  = np.clip((cost_chip - vmin) / (vmax - vmin + 1e-6), 0, 1)
        inv_cost   = 1.0 - cost_norm     # high inside, low at boundary

        # ---------------- STACK & WRITE ------------------------------------
        comp_arr = np.stack([prob_s, chm_norm, inv_cost]).astype("float32")

        out_path = OUT_TILE_DIR / f"{prob_path.stem}_comp.tif"
        meta = prob_src.meta.copy()
        meta.update({"count": 3, "dtype": "float32", "nodata": np.nan})
        with rasterio.open(out_path, "w", **meta) as dst:
            for i in range(3):
                dst.write(comp_arr[i], i + 1)
        # -------------------------------------------------------------------

        # save centre point
        centre_records.append({"cluster": cluster_id,
                               "geometry": Point(centre_xy[0], centre_xy[1])})

# ---------------- WRITE CENTRE POINTS --------------------------------------
centre_gdf = gpd.GeoDataFrame(centre_records, crs=prob_src.crs)
if OUT_CENTRES_GPKG.exists():
    os.remove(OUT_CENTRES_GPKG)
centre_gdf.to_file(OUT_CENTRES_GPKG, driver="GPKG")
print(f"✓  wrote {len(centre_gdf)} composite tiles to {OUT_TILE_DIR}")
print(f"✓  wrote centre points to {OUT_CENTRES_GPKG}")