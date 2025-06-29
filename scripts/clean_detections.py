#!/usr/bin/env python3
"""
clean_detections.py
--------------------------------------------------------------
1.  Delete polygons that strictly contain another polygon
2.  Delete polygons outside [MIN_AREA_M2, MAX_AREA_M2] bounds
3.  Delete polygons whose underlying mask pixels are ≥90 % ground (0)
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely import __version__ as shapely_version
from packaging.version import parse
from rasterstats import zonal_stats            # pip install rasterstats
import rasterio

# ───── USER SETTINGS ───────────────────────────────────────────────
IN_GPKG        = Path("/Users/iosefa/repos/obia/docs/example_data/detections.gpkg")
MASK_RASTER    = Path("/Users/iosefa/repos/obia/docs/example_data/mask_full.tif")  # 0 = ground, 1 = canopy
OUT_GPKG       = Path("/Users/iosefa/repos/obia/docs/example_data/detections_clean.gpkg")
LAYER          = "detections"

MAX_AREA_M2    = 30 * 30      # 900
MIN_AREA_M2    = 2 * 2        #   4
GROUND_THRESH  = 0.9          # ≥ 90 % ground → drop
# ───────────────────────────────────────────────────────────────────

print(f"Reading {IN_GPKG.name} …")
gdf = gpd.read_file(IN_GPKG, layer=LAYER).reset_index(drop=True)
print(f"Loaded {len(gdf):,} polygons")

# ── STEP 1 ▸ drop outer boxes --------------------------------------
predicate = "contains_properly" if parse(shapely_version) >= parse("2.0") else "contains"
gdf["__id"] = gdf.index
pairs = gpd.sjoin(
    gdf[["__id", "geometry"]], gdf[["__id", "geometry"]],
    predicate=predicate, how="inner",
)
outer_ids = pairs.query("__id_left != __id_right")["__id_left"].unique()
gdf = gdf[~gdf["__id"].isin(outer_ids)].drop(columns="__id")
print(f"Removed {len(outer_ids):,} outer boxes")

# ── STEP 2 ▸ filter by area ----------------------------------------
if gdf.crs.is_geographic:
    lon, lat = gdf.geometry.unary_union.centroid.coords[0]
    utm = f"+proj=utm +zone={int((lon + 180)//6 + 1)} +datum=WGS84 +units=m +no_defs"
    gdf_proj = gdf.to_crs(utm)
else:
    gdf_proj = gdf
areas = gdf_proj.area
gdf = gdf[(areas < MAX_AREA_M2) & (areas > MIN_AREA_M2)]
print("After area filter:", len(gdf))

# ── STEP 3 ▸ filter by ground mask ---------------------------------
print("Evaluating canopy mask …")
with rasterio.open(MASK_RASTER) as src:
    if gdf.crs != src.crs:
        gdf_mask = gdf.to_crs(src.crs)
    else:
        gdf_mask = gdf.copy()

stats = zonal_stats(
    gdf_mask, MASK_RASTER, stats=["mean"], nodata=0, geojson_out=False,
    all_touched=False, raster_out=False,
)

# mask is binary 0/1 → mean = canopy-fraction
canopy_frac = [s["mean"] if s["mean"] is not None else 0.0 for s in stats]
ground_mask = [(1 - cf) >= GROUND_THRESH for cf in canopy_frac]
dropped = sum(ground_mask)
gdf = gdf[~pd.Series(ground_mask).values]
print(f"Removed {dropped:,} ground polygons (≥{int(GROUND_THRESH*100)} % ground)")

# ── SAVE ------------------------------------------------------------
gdf.to_file(OUT_GPKG, layer=LAYER, driver="GPKG")
print(f"✓ Saved {len(gdf):,} cleaned polygons → {OUT_GPKG}")