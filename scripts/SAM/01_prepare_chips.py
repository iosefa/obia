#!/usr/bin/env python3
"""
Create IMG_SIZE×IMG_SIZE chips + masks for SAM fine‑tuning.

Inputs (same EPSG):
    image.tif   – 8‑band WorldView (bands numbered 1‑8)
    crowns.gpkg – polygon crowns

Outputs:
    data/chips/{img|msk|json}/<id>.npy|json
"""
import os, json, numpy as np, rasterio, geopandas as gpd
from pathlib import Path

from rasterio.features import rasterize
from shapely.geometry  import box

# ─── user paths ────────────────────────────────────────────────────
ROOT   = Path(__file__).parent           # folder SAM/
IMG    = ROOT / "image.tif"
VEC    = ROOT / "crowns.gpkg"
OUTDIR = ROOT / "data" / "chips"

IMG_SIZE  = 512
OVERLAP   = 480               # 32‑px stride
RGB_BANDS = (6, 4, 2)         # NIR‑1, Red, Green   (0‑based)

# ─── out dirs ──────────────────────────────────────────────────────
for sub in ("img", "msk", "json"):
    (OUTDIR / sub).mkdir(parents=True, exist_ok=True)

with rasterio.open(IMG) as src:
    rgb = np.stack([src.read(b+1) for b in RGB_BANDS])          # (3,H,W)
    rgb8 = ((rgb - rgb.min()) / (np.ptp(rgb) + 1e-9) * 255).astype("uint8")
    tr  = src.transform
    H, W = src.height, src.width
    rbox = box(*src.bounds)

crowns = gpd.read_file(VEC).to_crs(src.crs)
crowns = crowns[crowns.geometry.is_valid & crowns.intersects(rbox)]

mask = rasterize(
    [(g, 1) for g in crowns.geometry], out_shape=(H, W), transform=tr,
    fill=0, all_touched=True, dtype="uint8"
)
print("crown pixels:", int(mask.sum()))

stride, tid = IMG_SIZE - OVERLAP, 0
for y0 in range(0, H, stride):
    for x0 in range(0, W, stride):
        y1, x1 = min(y0+IMG_SIZE, H), min(x0+IMG_SIZE, W)
        chip = rgb8[:, y0:y1, x0:x1]
        msk  = mask [y0:y1, x0:x1]
        if msk.sum() == 0:
            continue
        # pad to full size (only borders)
        pad_h, pad_w = IMG_SIZE-chip.shape[1], IMG_SIZE-chip.shape[2]
        if pad_h or pad_w:
            chip = np.pad(chip, ((0,0),(0,pad_h),(0,pad_w)))
            msk  = np.pad(msk,  ((0,pad_h),(0,pad_w)))
        ys, xs = np.where(msk)
        json.dump(
            dict(point=[int(xs.mean()), int(ys.mean())],
                 box=[int(xs.min()), int(ys.min()),
                      int(xs.max()), int(ys.max())]),
            open(OUTDIR/"json"/f"{tid}.json","w")
        )
        np.save(OUTDIR/"img"/f"{tid}.npy", chip)
        np.save(OUTDIR/"msk"/f"{tid}.npy", msk)
        tid += 1
print(f"✓ {tid} chips ➜ {OUTDIR}")