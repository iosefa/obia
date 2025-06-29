#!/usr/bin/env python3
"""
make_tiles.py
-------------
Slice the full probability raster + instance-mask into 512×512 tiles.
Only tiles that actually contain crowns are kept.

Outputs:
train/x/tile_####.tif   (float32 0-1)
train/y/tile_####.tif   (uint16  instance IDs)
"""

import rasterio, os, tifffile as tiff
from rasterio.windows import Window
import numpy as np

IMG = "canopy_index_merged.tif"
MSK = "crowns_mask.tif"
TILE = 512         # tile size (you can change to 256 or 1024)

os.makedirs("train/x", exist_ok=True)
os.makedirs("train/y", exist_ok=True)

with rasterio.open(IMG) as src_i, rasterio.open(MSK) as src_m:
    H, W = src_i.height, src_i.width
    assert (H, W) == src_m.shape

    k = 0
    for y in range(0, H, TILE):
        for x in range(0, W, TILE):
            win = Window(x, y, min(TILE, W - x), min(TILE, H - y))
            img = src_i.read(1, window=win).astype("float32")
            msk = src_m.read(1, window=win).astype("uint16")

            if msk.max() == 0:          # skip empty tiles
                continue

            img = (img - img.min()) / (img.ptp() + 1e-8)  # normalise 0-1
            tiff.imwrite(f"train/x/tile_{k:04d}.tif", img, dtype="float32")
            tiff.imwrite(f"train/y/tile_{k:04d}.tif", msk, dtype="uint16")
            k += 1

print(f"✓ wrote {k} training tile pairs in train/x & train/y")