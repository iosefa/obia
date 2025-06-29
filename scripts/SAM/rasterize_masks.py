#!/usr/bin/env python3
"""
clean_and_rasterise_crowns.py
─────────────────────────────
• Repair invalid geometries.
• Remove overlaps (largest crowns win).
• Write cleaned crowns and a uint16 label mask aligned to a reference raster.
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid              # Shapely ≥ 2.x

# ──── PATHS ────────────────────────────────────────────────────────────────
RASTER_REF       = Path("canopy_index.tif")
CROWNS_RAW_GPKG  = Path("crowns_train.gpkg")
MASK_OUT_TIF     = Path("crowns_mask.tif")
# --------------------------------------------------------------------------

print("• reading crowns …")
gdf_raw = gpd.read_file(CROWNS_RAW_GPKG)
crs     = gdf_raw.crs

# ─── 1.  repair invalid geometries ────────────────────────────────────────
def repair(geom):
    if geom.is_valid:
        return geom
    try:
        fixed = make_valid(geom)        # Shapely 2 way
    except Exception:
        fixed = geom.buffer(0)          # fallback
    # keep only Polygon/MultiPolygon results (safest for rasterise)
    if isinstance(fixed, (Polygon, MultiPolygon)) and not fixed.is_empty:
        return fixed
    return None

gdf_raw["geometry"] = gdf_raw.geometry.apply(repair)
n_repaired = gdf_raw["geometry"].isna().sum()
if n_repaired:
    print(f"   ⚠ removed {n_repaired} totally irreparable crowns")

gdf_raw = gdf_raw.dropna(subset=["geometry"]).reset_index(drop=True)
gdf_raw["area"] = gdf_raw.geometry.area
print(f"   {len(gdf_raw)} crowns remain after repair")

# ─── 2.  remove overlaps (largest → smallest) ──────────────────────────────
accepted, cleaned_rows = [], []
for _, row in gdf_raw.sort_values("area", ascending=False).iterrows():
    geom_left = row.geometry
    if accepted:
        geom_left = geom_left.difference(unary_union(accepted))
    if not geom_left.is_empty:
        accepted.append(geom_left)
        cleaned_rows.append({"geometry": geom_left})

gdf_clean = gpd.GeoDataFrame(cleaned_rows, crs=crs)
gdf_clean["id"] = np.arange(1, len(gdf_clean) + 1, dtype=np.uint16)

print(f"   kept {len(gdf_clean)} crowns (non-overlapping)")

# ─── 3.  rasterise cleaned crowns ─────────────────────────────────────────
with rasterio.open(RASTER_REF) as ref:
    meta = ref.meta.copy()
    meta.update(count=1, dtype="uint16", compress="lzw", nodata=0)

    shapes = ((geom, val) for geom, val in zip(gdf_clean.geometry, gdf_clean["id"]))
    mask = rasterize(
        shapes        = shapes,
        out_shape     = ref.shape,
        transform     = ref.transform,
        fill          = 0,
        dtype         = "uint16",
        all_touched   = False           # stricter: centre-in-polygon only
    )

    with rasterio.open(MASK_OUT_TIF, "w", **meta) as dst:
        dst.write(mask, 1)

print(f"✓ wrote label mask ({len(gdf_clean)} crowns) → {MASK_OUT_TIF}")