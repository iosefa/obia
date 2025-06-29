#!/usr/bin/env python3
"""
prune_df_edgeboxes.py
────────────────────────────────────────────────────────────────────
1. Build a fish-net grid that mirrors the tiling used for DeepForest.
2. Save that grid to disk (useful for QA).
3. Remove every DeepForest detection whose *entire* box touches the
   outer rim of its tile (rim width = SHRINK).

No transforms.json required.
"""

# ---------- USER PATHS --------------------------------------------------------
RASTER          = "/Users/iosefa/repos/obia/docs/example_data/image_full.tif"
DETECTIONS_GPKG = "/Users/iosefa/repos/obia/docs/example_data/detections_clean.gpkg"
LAYER_IN        = "detections"

FISHNET_GPKG    = "/Users/iosefa/repos/obia/docs/example_data/tiles_fishnet.gpkg"
PRUNED_GPKG     = "/Users/iosefa/repos/obia/docs/example_data/detections_pruned.gpkg"
LAYER_OUT       = "detections"

TILE_SIZE_M = 179.2     # 512 px × 0.35 m
OVERLAP_M   = 20.0
SHRINK      = 0.01      # delete this rim width (metres)
# ------------------------------------------------------------------------------

from pathlib import Path
import geopandas as gpd
import rasterio
from shapely.geometry import box
from rasterio.coords import BoundingBox

# ------------------------------------------------------------------
# 1 ▸ build the fish-net grid  (bottom-up, same as generate_tiles)
# ------------------------------------------------------------------
step = TILE_SIZE_M - OVERLAP_M
with rasterio.open(RASTER) as src:
    b: BoundingBox = src.bounds
    crs_raster = src.crs

tiles = []
row = 0
y_bot = b.bottom
while y_bot < b.top:
    col = 0
    x_left = b.left
    y_top = y_bot + TILE_SIZE_M
    while x_left < b.right:
        tiles.append(
            {
                "tile_id": f"r{row}_c{col}",
                "geometry": box(
                    x_left,
                    y_bot,
                    x_left + TILE_SIZE_M,
                    y_top,
                ),
            }
        )
        x_left += step
        col += 1
    y_bot += step
    row += 1

tiles_gdf = gpd.GeoDataFrame(tiles, crs=crs_raster)
tiles_gdf["rim_inset"] = tiles_gdf.geometry.buffer(-SHRINK)

# save fish-net (retain just one geometry column)
tiles_gdf[["tile_id", "geometry"]].to_file(
    FISHNET_GPKG, layer="tiles", driver="GPKG", overwrite=True
)
print(f"[✓] fish-net saved → {FISHNET_GPKG}   ({len(tiles_gdf):,} tiles)")

# ------------------------------------------------------------------ #
# 2 ▸ load detections & drop edge-touchers                           #
# ------------------------------------------------------------------ #
dets = gpd.read_file(DETECTIONS_GPKG, layer=LAYER_IN)

# harmonise CRS
if dets.crs != tiles_gdf.crs:
    tiles_gdf = tiles_gdf.to_crs(dets.crs)

# inner rims as a proper GeoDataFrame
inner_gdf = (
    tiles_gdf[["tile_id", "rim_inset"]]
    .rename(columns={"rim_inset": "geometry"})
    .set_geometry("geometry")
)

# --- shortlist detections whose centroids lie inside an inner rim --
centroids = dets.copy()
centroids["geometry"] = centroids.geometry.centroid

join_idx = gpd.sjoin(
    centroids,
    inner_gdf,
    predicate="within",
    how="left",
)

# If a centroid hits multiple inner rims, keep the *first* one
tile_map = (
    join_idx.dropna(subset=["tile_id"])
            .groupby(level=0)["tile_id"]
            .first()
)

# --- keep only boxes fully contained inside the corresponding rim --
keep_idx = []
# turn inner_gdf into a dict for fast lookup
rim_dict = dict(zip(inner_gdf["tile_id"], inner_gdf.geometry))

for idx, t_id in tile_map.items():
    rim_poly = rim_dict[t_id]
    if dets.at[idx, "geometry"].within(rim_poly):
        keep_idx.append(idx)

kept = dets.loc[keep_idx]

print(f"Kept {len(kept):,} / {len(dets):,} DeepForest boxes")

kept.to_file(PRUNED_GPKG, layer=LAYER_OUT, driver="GPKG", overwrite=True)
print(f"✓ pruned detections → {PRUNED_GPKG}")