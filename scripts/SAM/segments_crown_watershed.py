#!/usr/bin/env python
"""
Improved seed‑guided watershed for tree‑crown segmentation
----------------------------------------------------------

Changes vs. v1
  • a *blob* (all pixels with P ≥ PROB_SEED) becomes the marker for each seed
  • CHM peaks add backup markers where you had no probability raster
  • overlapping blobs are resolved by "winner‑takes‑highest‑P"
  • everything else (cost‑surface flood‑fill, small‑crown filter) unchanged
"""
import glob, os, sys, itertools
import numpy as np
import rasterio
from rasterio.enums import Resampling
from skimage.segmentation import watershed     # segmentation solver
from skimage.feature import peak_local_max     # CHM peaks
from skimage.measure import label              # connected components
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# ───────────────────────────────────────────── editable parameters ────────── #
COST_PATH   = "cost.tif"
CHM_PATH    = "chm.tif"
PROB_DIR    = "probs"              # folder with per‑seed probability TIFFs
OUT_PATH    = "crowns.tif"

PROB_SEED   = 0.2                 # ≥ this value → part of marker blob
PROB_MIN    = 0.15                 # if *max* prob < this → ignore raster
CHM_MIN_H   = 2.0                  # m; CHM mask & backup peak search
CHM_PEAK_D  = 3                    # pixels; minima distance between peaks
MIN_SIZE_PX = 80                   # remove crowns < this area afterwards
# ──────────────────────────────────────────────────────────────────────────── #

def read_band_like(path, ref):
    with rasterio.open(path) as src:
        if (src.transform != ref.transform) or (src.shape != ref.shape):
            band = src.read(
                1,
                out_shape=(ref.height, ref.width),
                resampling=Resampling.bilinear,
            )
        else:
            band = src.read(1)
    return band.astype(np.float32)

# 1 ─ reference rasters -------------------------------------------------------
with rasterio.open(COST_PATH) as cost_src:
    cost      = cost_src.read(1).astype(np.float32)
    profile   = cost_src.profile.copy()
    H, W      = cost.shape

chm = read_band_like(CHM_PATH, cost_src)

canopy_mask = chm >= CHM_MIN_H

# 2 ─ probability rasters → marker blobs -------------------------------------
prob_files = sorted(glob.glob(os.path.join(PROB_DIR, "*.tif")))
if not prob_files:
    sys.exit(f"No probability rasters found in {PROB_DIR}")

markers      = np.zeros((H, W), dtype=np.int32)
prob_at_mark = np.zeros((H, W), dtype=np.float32)  # keep max‑P per pixel
next_id      = 1

print("Converting probability rasters to marker blobs …")
for path in tqdm(prob_files):
    prob = read_band_like(path, cost_src)

    # skip rasters with very weak evidence everywhere
    if prob.max() < PROB_MIN:
        continue

    blob = prob >= PROB_SEED
    blob &= canopy_mask                    # forbid ground pixels

    if not blob.any():
        # fallback: take the global maximum pixel
        y, x = np.unravel_index(prob.argmax(), prob.shape)
        if canopy_mask[y, x]:
            blob[y, x] = True
        else:
            continue                       # still unusable

    # resolve overlaps: pixel keeps the label of *highest* probability seen
    overwrite = blob & (prob > prob_at_mark)
    markers[overwrite] = 0                 # clear previous label if any
    markers[blob & overwrite] = next_id
    prob_at_mark[overwrite] = prob[overwrite]

    next_id += 1

print(f"Markers after probabilities  : {next_id-1} labels")

# 3 ─ CHM local maxima fill gaps ---------------------------------------------
# Where there is canopy but zero marker, drop extra seeds at height peaks.
need_mark  = canopy_mask & (markers == 0)
if need_mark.any():
    print("Adding backup markers from CHM peaks …")
    # distance map limits one peak / CHM_PEAK_D px
    peaks = peak_local_max(
        chm, labels=need_mark.astype(np.uint8), min_distance=CHM_PEAK_D,
        threshold_abs=CHM_MIN_H
    )
    for y, x in peaks:
        markers[y, x] = next_id
        next_id += 1

print(f"Total markers after CHM peaks: {next_id-1}")

if next_id == 1:
    sys.exit("No valid marker pixels – tune thresholds!")

# 4 ─ prepare the 'topography' (low = valleys grow first) --------------------
topo = cost.copy()
topo -= topo.min()
rng = topo.max()
topo /= rng if rng else 1.0                 # 0‑1 range

# 5 ─ watershed --------------------------------------------------------------
print("Running watershed …")
labels_ws = watershed(topo, markers=markers, mask=canopy_mask,
                      compactness=0.0).astype(np.uint32)

print(f"Watershed produced {labels_ws.max()} raw crowns")

# 6 ─ drop tiny fragments ----------------------------------------------------
if MIN_SIZE_PX:
    counts = np.bincount(labels_ws.ravel())
    tiny   = np.where((counts < MIN_SIZE_PX) & (np.arange(len(counts)) != 0))[0]
    if tiny.size:
        print(f"Removing {tiny.size} crowns < {MIN_SIZE_PX} px")
        labels_ws[np.isin(labels_ws, tiny)] = 0

# 7 ─ write GeoTIFF ----------------------------------------------------------
profile.update(dtype=rasterio.uint32, count=1,
               compress="deflate", predictor=2)
with rasterio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(labels_ws, 1)

print(f"✓ saved {OUT_PATH} with {labels_ws.max()} crowns (IDs 1…N)")