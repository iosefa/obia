#!/usr/bin/env python3
"""
02_manual_feats.py
────────────────────────────────────────────────────────────────────
Compute OBIA-style descriptors (spectral, textural, LiDAR structural,
LiDAR radiometric) for each pre-labelled crown polygon.

Inputs (site-1 directory)
-------------------------
crowns_labeled.gpkg          ← from 01_labeled_crowns.py
image_full.tif               ← 9-band WV-3 mosaic   (shape: H × W × 9)
ept/ept.json                 ← airborne-LiDAR point cloud in EPT format

Output
------
crowns_manual_feats.gpkg     : one row per crown, + dozens of feature columns
"""

from pathlib import Path

import numpy as np
import geopandas as gpd

from obia.handlers.geotif import open_geotiff
from obia.segmentation.segment_statistics import create_objects

# ------------------------------------------------------------------
# paths (edit if your folder names differ)
# ------------------------------------------------------------------
BASE_DIR       = Path("/Users/iosefa/repos/obia/docs/example_data/site_1")

RASTER_PATH    = BASE_DIR / "image.tif"        # 9-band WV-3 mosaic
CROWNS_GPKG    = BASE_DIR / "crowns_labeled.gpkg"   # from step 01
EPT_PATH       = BASE_DIR / "ept/ept.json"          # LAZ ↓ to EPT
EPT_SRS        = "EPSG:32605"                       # CRS of the LiDAR

OUT_GPKG       = BASE_DIR / "crowns_manual_feats.gpkg"

# ------------------------------------------------------------------
# 1 ▸ open the imagery
# ------------------------------------------------------------------
print("• opening WorldView-3 cube …")
wv = open_geotiff(str(RASTER_PATH))

# open_geotiff returns (H, W, B); transpose to (B, H, W) for mobia
wv.img_data = np.transpose(wv.img_data, (2, 0, 1))
print("  raster shape (bands, H, W):", wv.img_data.shape)

# ------------------------------------------------------------------
# 2 ▸ read labelled crowns
# ------------------------------------------------------------------
crowns = gpd.read_file(CROWNS_GPKG)
print(f"• crowns loaded : {len(crowns):,}")

# ------------------------------------------------------------------
# 3 ▸ feature extraction
# ------------------------------------------------------------------
print("• computing object descriptors …")
objects_gdf = create_objects(
    segments=crowns,
    image=wv,
    ept=str(EPT_PATH),
    ept_srs=EPT_SRS,
    textural_bands=[0],            # use coastal-blue for Haralick textures
    voxel_resolution=(500, 500, 2),   # spatialXY=0.5 m × 0.5 m, Z=2 m
    calculate_structural=True,
    calculate_radiometric=True,
)

print("  feature columns:", [c for c in objects_gdf.columns if c not in crowns.columns])

# ------------------------------------------------------------------
# 4 ▸ write output
# ------------------------------------------------------------------
objects_gdf.to_file(OUT_GPKG, driver="GPKG", overwrite=True)
print("✓ wrote", OUT_GPKG)