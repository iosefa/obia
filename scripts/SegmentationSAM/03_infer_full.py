#!/usr/bin/env python3
"""
03_infer_full.py  –  apply your tuned SAM‑ViT‑L decoder to the full scene
Produces:
    crowns_sam_mask.tif   (8‑bit crown mask)
    crowns_sam.gpkg       (vector polygons, EPSG:6635)
"""

import numpy as np, rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon
import geopandas as gpd, torch
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ───────── paths ───────────────────────────────────────────────────
RASTER     = "train_img.tif"
WEIGHTS    = "sam_crowns_vit_l_decoder.pth"
MODEL_TYPE = "vit_l"

# output
MASK_TIF   = "crowns_sam_mask.tif"
VEC_GPKG   = "crowns_sam.gpkg"

# ───────── load tuned model ────────────────────────────────────────
device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[MODEL_TYPE](checkpoint=WEIGHTS).to(device).eval()

# one mask per image (matches training)
sam.mask_decoder.num_multimask_outputs = 1

mask_gen = SamAutomaticMaskGenerator(
    model=sam,
    pred_iou_thresh=0.4,
    stability_score_thresh=0.88,
    min_mask_region_area=200,      # pixel area threshold
    points_per_side=None           # prompt‑free
)

# ───────── prepare output raster ───────────────────────────────────
with rasterio.open(RASTER) as src:
    profile = src.profile
    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    full_mask = np.zeros((src.height, src.width), dtype="uint8")

    # build 3‑band composite once (NIR‑1, Red, Green → bands 7,5,3)
    bands = src.read([7,5,3])                       # (3,H,W)
    bands = ((bands - bands.min()) /
             (bands.ptp() + 1e-9) * 255).astype("uint8")

    TILE  = 1024
    OVER  = 512
    for y0 in range(0, src.height, TILE-OVER):
        for x0 in range(0, src.width, TILE-OVER):
            h = min(TILE, src.height - y0)
            w = min(TILE, src.width  - x0)

            chip = bands[:, y0:y0+h, x0:x0+w]
            # pad to 1024×1024
            chip_pad = np.zeros((3, TILE, TILE), dtype="uint8")
            chip_pad[:, :h, :w] = chip

            masks = mask_gen.generate(chip_pad.transpose(1,2,0))
            for m in masks:
                if m["area"] < 200:          # extra safety filter
                    continue
                ys, xs = np.where(m["segmentation"])
                ys = ys + y0
                xs = xs + x0
                full_mask[ys, xs] = 1

# write crown mask raster
with rasterio.open(MASK_TIF, "w", **profile) as dst:
    dst.write(full_mask, 1)

print(f"✓ crown mask written → {MASK_TIF}")

# ───────── vectorise mask to polygons ──────────────────────────────
polys = []
for vals, shapes in rasterio.features.shapes(full_mask, mask=full_mask==1,
                                             transform=profile["transform"]):
    geom = Polygon(shapes["coordinates"][0])
    if geom.is_valid and geom.area > profile["transform"].a**2:   # >1 pixel
        polys.append(geom)

gdf = gpd.GeoDataFrame(geometry=polys, crs=profile["crs"])
gdf.to_file(VEC_GPKG, driver="GPKG")
print(f"✓ vector crowns written → {VEC_GPKG}")