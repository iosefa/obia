#!/usr/bin/env python3
"""
Merge seed_*_comp.tif tiles into a single-band canopy-likelihood raster
with a *soft* ground-smoothing term.

Index  =  base_index · ground_weight^W_GROUND
where   base_index = prob · (1−W_CHM + W_CHM·chm_norm) · (1−W_COST + W_COST·(1−cost))
        ground_weight = 1 / (1 + exp(−(chm − TH)/SIGMA))
"""

import glob, os, numpy as np, rasterio
from rasterio.merge import merge

# ─── user-tuneable parameters ────────────────────────────────────────────
TILE_DIR   = "composite_tiles_ns2"
OUT_RASTER = "canopy_index2.tif"

CHM_MAX = 25.0        # m mapping to chm_norm = 1
W_CHM   = 0.75         # CHM influence in base index
W_COST  = 0.25         # cost influence in base index

GROUND_THRESH = 3.0   # m height where ground_weight ≈ 0.5
SIGMA         = 2.0   # m, broader ⇒ softer transition
W_GROUND      = 1   # 0 = ignore ground term, 1 = full strength

NODATA = 0      # sentinel NoData
# ─────────────────────────────────────────────────────────────────────────

# ─── collect tileså ───────────────────────────────────────────────────────
paths = sorted(glob.glob(f"{TILE_DIR}/seed_*_comp.tif"))
if not paths:
    raise FileNotFoundError(f"No seed_*_comp.tif files in {TILE_DIR}")
srcs = [rasterio.open(p) for p in paths]

def mosaic_band(idx):
    data, transform = merge(
        srcs,
        indexes=[idx],
        dtype="float32",
        nodata=NODATA,
        method="max",
    )
    return data[0], transform

prob, out_transform = mosaic_band(1)
chm,  _             = mosaic_band(2)
cost, _             = mosaic_band(3)

# ─── base index (same as before) ────────────────────────────────────────
chm_norm = np.clip(chm / CHM_MAX, 0, 1)
inv_cost = 1 - cost
base     = prob * (1 - W_CHM + W_CHM * chm_norm) * (1 - W_COST + W_COST * inv_cost)

# ─── soft ground-weight ─────────────────────────────────────────────────
ground_weight = 1.0 / (1.0 + np.exp(-(chm - GROUND_THRESH) / SIGMA))
# raise ground_weight to W_GROUND to make effect softer when W_GROUND < 1
smooth_weight = ground_weight ** W_GROUND

index = base * smooth_weight

# ─── handle NoData & rescale 0-1 ────────────────────────────────────────
mask  = (prob == NODATA) | (chm == NODATA) | (cost == NODATA)
index = np.where(mask, np.nan, index)

vmin, vmax = np.nanmin(index), np.nanmax(index)
index = np.where(
    np.isnan(index),
    NODATA,
    (index - vmin) / (vmax - vmin) if vmax > vmin else 0
)

# ─── write output ───────────────────────────────────────────────────────
meta = srcs[0].meta.copy()
meta.update({
    "driver": "GTiff",
    "height": index.shape[0],
    "width" : index.shape[1],
    "count" : 1,
    "dtype" : "float32",
    "nodata": NODATA,
    "transform": out_transform,
})

with rasterio.open(OUT_RASTER, "w", **meta) as dst:
    dst.write(index, 1)

print(f"✓ Wrote softer canopy index → {OUT_RASTER}")

for s in srcs:
    s.close()


# #!/usr/bin/env python3
# """
# Merge seed_*_comp.tif tiles into a single-band canopy-likelihood raster.
#
#   • Band-1: probability (0-1)
#   • Band-2: CHM height (metres)       → scaled to 0-1 by CHM_MAX
#   • Band-3: cost surface (0-1), high = canopy break
#
# Index  =  prob
#          * (1 - W_CHM  + W_CHM  * chm_norm)
#          * (1 - W_COST + W_COST * (1 - cost))
#
# Then rescale valid pixels to 0-1.
#
# Pixels that are NoData in **any** input band remain NoData.
# Overlaps across tiles are averaged before the index is computed.
# """
#
# import glob, os, numpy as np, rasterio
# from rasterio.merge import merge
#
# # ─── tweakables ──────────────────────────────────────────────────────────
# TILE_DIR   = "composite_tiles_ns"
# OUT_RASTER = "canopy_index_merged.tif"
#
# CHM_MAX = 20.0      # metres that map to chm_norm = 1
# W_CHM  = 0.7        # 0 → ignore CHM, 1 → full weight
# W_COST = 0.3        # 0 → ignore cost, 1 → full weight
#
# NODATA  = -9999.0   # sentinel outside [0,1] range; stays transparent
# # ─────────────────────────────────────────────────────────────────────────
#
# # ─── gather tiles ────────────────────────────────────────────────────────
# paths = sorted(glob.glob(f"{TILE_DIR}/seed_*_comp.tif"))
# if not paths:
#     raise FileNotFoundError(f"No seed_*_comp.tif files found under {TILE_DIR}")
# srcs = [rasterio.open(p) for p in paths]
#
# # ─── helper to mosaic a single band ──────────────────────────────────────
# def mosaic_band(band_idx: int):
#     data, transform = merge(
#         srcs,
#         indexes=[band_idx],
#         dtype="float32",
#         nodata=NODATA,
#         method="max",           # mean of valid samples where overlaps
#     )
#     return data[0], transform       # strip the leading band dimension
#
# # ─── build mosaics ───────────────────────────────────────────────────────
# prob, out_transform = mosaic_band(1)
# chm,  _             = mosaic_band(2)
# cost, _             = mosaic_band(3)
#
# # ─── compute the index ───────────────────────────────────────────────────
# chm_norm = np.clip(chm / CHM_MAX, 0, 1)   # 0-1 height
# inv_cost = 1 - cost                       # 1 = interior of crown
#
# index = prob \
#         * (1 - W_CHM  + W_CHM  * chm_norm) \
#         * (1 - W_COST + W_COST * inv_cost)
#
# # mask nodata as NaN for easier statistics
# mask = (prob == NODATA) | (chm == NODATA) | (cost == NODATA)
# index = np.where(mask, np.nan, index)
#
# # ─── rescale valid pixels to 0-1 ─────────────────────────────────────────
# valid_min = np.nanmin(index)
# valid_max = np.nanmax(index)
#
# if valid_max > valid_min:
#     index = (index - valid_min) / (valid_max - valid_min)
# else:
#     index = np.zeros_like(index)
#
# # restore sentinel NoData
# index = np.where(np.isnan(index), NODATA, index)
#
# # ─── write output raster ────────────────────────────────────────────────
# meta = srcs[0].meta.copy()
# meta.update({
#     "driver"   : "GTiff",
#     "height"   : index.shape[0],
#     "width"    : index.shape[1],
#     "count"    : 1,
#     "dtype"    : "float32",
#     "nodata"   : NODATA,
#     "transform": out_transform,
# })
#
# with rasterio.open(OUT_RASTER, "w", **meta) as dst:
#     dst.write(index, 1)
#
# print(f"✓ Wrote canopy index → {OUT_RASTER}")
#
# # ─── close sources ──────────────────────────────────────────────────────
# for s in srcs:
#     s.close()
#
#
# # #!/usr/bin/env python3
# # """
# # Merge probability tiles (band-1 of seed_*_comp.tif) so that
# #   • NoData stays transparent (doesn't clobber real data)
# #   • overlapping pixels are averaged (mean of valid values)
# #
# # Usage:  python merge_prob_tiles.py
# # """
# #
# # import glob, os, numpy as np, rasterio
# # from rasterio.merge import merge
# #
# # # ─── paths ───────────────────────────────────────────────────────────────
# # TILE_DIR   = "composite_tiles_ns"
# # OUT_RASTER = "prob_surface_merged_ns.tif"
# #
# # paths = sorted(glob.glob(f"{TILE_DIR}/seed_*_comp.tif"))
# # if not paths:
# #     raise FileNotFoundError(f"No seed_*_comp.tif files found under {TILE_DIR}")
# #
# # # ─── choose a nodata placeholder that can never be a real probability ────
# # NODATA = -9999.0               # outside 0–1 range, safe sentinel
# #
# # # ─── open all tiles ------------------------------------------------------
# # srcs = [rasterio.open(p) for p in paths]
# #
# # # ─── build the mosaic ----------------------------------------------------
# # #   • indexes=[1]      → first band only
# # #   • method='average' → average of valid pixels where tiles overlap
# # #   • fill_value=NODATA & nodata=NODATA ensure masked arithmetic:
# # #     any pixel with value == NODATA is ignored in the averaging
# # mosaic, out_transform = merge(
# #     srcs,
# #     indexes=[1],
# #     dtype="float32",
# #     nodata=NODATA,
# #     method="max"
# # )                                     # mosaic shape → (1, rows, cols)
# #
# # # ─── write result --------------------------------------------------------
# # meta = srcs[0].meta.copy()
# # meta.update({
# #     "driver"   : "GTiff",
# #     "height"   : mosaic.shape[1],
# #     "width"    : mosaic.shape[2],
# #     "count"    : 1,                   # single band
# #     "dtype"    : "float32",
# #     "nodata"   : NODATA,
# #     "transform": out_transform,
# # })
# #
# # with rasterio.open(OUT_RASTER, "w", **meta) as dst:
# #     dst.write(mosaic[0], 1)           # strip the band dimension
# #
# # print(f"✓ Wrote merged probability surface → {OUT_RASTER}")
# #
# # # ─── clean up ------------------------------------------------------------
# # for s in srcs:
# #     s.close()
