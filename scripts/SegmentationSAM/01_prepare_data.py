#!/usr/bin/env python3
"""
01_prepare_data.py  –  make SAM training chips from WorldView image + crown polygons
───────────────────────────────────────────────────────────────────────────────────
* Takes:  train_img.tif  (8-band, EPSG:6635)
          train_crowns.gpkg  (manual crown polygons, same CRS)
* Produces:   data/chips/{img|msk|json}/<id>.npy|json  – exactly IMG_SIZE×IMG_SIZE
"""

import os, json, numpy as np, rasterio, geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize

# ── paths & constants ───────────────────────────────────────────────
IMG  = "/Users/iosefa/repos/obia/scripts/SegmentationSAM/train_img.tif"
VEC  = "/Users/iosefa/repos/obia/scripts/SegmentationSAM/train_crowns.gpkg"
LAYER = None                         # set to layer name if GPKG has >1

OUT = "data"
IMG_SIZE = 512                       # chip edge in pixels
OVERLAP  = 480                        # pixels of overlap
RGB_BANDS = (6, 4, 2)                # WV-3 (NIR-1, Red, Green) – 0-based

# ── output folders ──────────────────────────────────────────────────
for sub in ("img", "msk", "json"):
    os.makedirs(f"{OUT}/chips/{sub}", exist_ok=True)

# ── load raster & make 3-band uint8 composite ───────────────────────
with rasterio.open(IMG) as src:
    rgb = np.stack([src.read(b + 1) for b in RGB_BANDS])          # (3,H,W)
    rgb8 = ((rgb - rgb.min()) / (np.ptp(rgb) + 1e-9) * 255).astype("uint8")
    transform = src.transform
    height, width = src.height, src.width
    rbox = box(*src.bounds)

# ── load & validate crowns ──────────────────────────────────────────
crowns = gpd.read_file(VEC, layer=LAYER)
if crowns.empty:
    raise ValueError("No polygons loaded – check VEC path or layer name")

if crowns.crs != src.crs:
    crowns = crowns.to_crs(src.crs)

crowns = crowns[crowns.geometry.is_valid]
crowns = crowns[crowns.intersects(rbox)]
if crowns.empty:
    raise ValueError("No crowns intersect raster extent – check input crop")

# ── rasterise crowns to mask ────────────────────────────────────────
mask = rasterize(
    [(geom, 1) for geom in crowns.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=True,
)
print("Total crown pixels burned:", int(mask.sum()))

# ── sliding-window tiling (inclusive of right/bottom edges) ─────────
stride  = IMG_SIZE - OVERLAP
tile_id = 0

for y0 in range(0, height, stride):
    for x0 in range(0, width, stride):
        y1 = min(y0 + IMG_SIZE, height)
        x1 = min(x0 + IMG_SIZE, width)

        chip = rgb8[:, y0:y1, x0:x1]
        msk  =  mask[y0:y1, x0:x1]

        if msk.sum() == 0:                         # skip empty tiles
            continue

        # pad edge tiles so every chip is IMG_SIZE×IMG_SIZE
        pad_h = IMG_SIZE - chip.shape[1]
        pad_w = IMG_SIZE - chip.shape[2]
        if pad_h or pad_w:
            chip = np.pad(chip, ((0,0), (0,pad_h), (0,pad_w)), mode="constant")
            msk  = np.pad(msk,  ((0,pad_h), (0,pad_w)),       mode="constant")

        # prompts: centroid point & tight bbox (pixel coords)
        ys, xs = np.where(msk)
        cy, cx = int(ys.mean()), int(xs.mean())
        bbox   = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        np.save(f"{OUT}/chips/img/{tile_id}.npy",  chip)
        np.save(f"{OUT}/chips/msk/{tile_id}.npy",  msk)
        json.dump(
            {"point": [cx, cy], "box": bbox, "h": IMG_SIZE, "w": IMG_SIZE},
            open(f"{OUT}/chips/json/{tile_id}.json", "w"),
        )
        tile_id += 1

print(f"✓ Prepared {tile_id} training chips ➜ {OUT}/chips")