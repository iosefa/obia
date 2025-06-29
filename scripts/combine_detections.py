#!/usr/bin/env python3
"""
combine_detections.py
Merge many single-tile detection GeoPackages into one.

Assumes each *.gpkg in DETECTIONS_DIR contains exactly one layer
(the default created by obia.utils.utils.save_deepforest_predictions_to_gpkg).

If CRS differ, everything is re-projected to the CRS of the first file.
"""

import glob
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ── EDIT THESE PATHS ────────────────────────────────────────────────
DETECTIONS_DIR = Path("/Users/iosefa/repos/obia/docs/example_data/detections")
OUTPUT_GPKG    = Path("/Users/iosefa/repos/obia/docs/example_data/detections.gpkg")
LAYER_NAME     = "detections"           # the layer inside the output GPKG
# ────────────────────────────────────────────────────────────────────

print(f"Scanning {DETECTIONS_DIR} …")
files = sorted(DETECTIONS_DIR.glob("*.gpkg"))
if not files:
    raise SystemExit("No *.gpkg files found – did step2.py run?")

gdfs = []
for fp in files:
    gdf = gpd.read_file(fp)
    gdfs.append(gdf)
    print("  +", fp.name, "→", len(gdf), "features")

# -- unify CRS (convert everything to first file’s CRS) -------------
target_crs = gdfs[0].crs
for i, gdf in enumerate(gdfs):
    if gdf.crs != target_crs:
        gdfs[i] = gdf.to_crs(target_crs)

# -- concatenate & write -------------------------------------------
merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=target_crs)
merged.to_file(OUTPUT_GPKG, layer=LAYER_NAME, driver="GPKG")
print(f"\n✓  Wrote {len(merged):,} total detections → {OUTPUT_GPKG}")