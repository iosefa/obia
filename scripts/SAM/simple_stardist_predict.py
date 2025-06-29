#!/usr/bin/env python3
"""
simple_stardist_predict.py
———————————
• Loads your merged probability raster
• Loads StarDist model (expects models/crown_stardist/*)
• Runs predict_instances_big with minimal arguments
• Saves:
    labels.tif     – uint16 label image (0 = background, 1… = crowns)
    crowns.gpkg    – polygons for each crown (no extra filtering)
"""

# --- user paths ---------------------------------------------------------
IMG_PATH   = "canopy_index_merged.tif"   # single-band, 0–1 floats
MODEL_BASE = "models"                    # parent of crown_stardist/
MODEL_NAME = "crown_stardist"
OUT_LABELS = "labels.tif"
OUT_VECT   = "crowns.gpkg"

# --- imports ------------------------------------------------------------
import rasterio, numpy as np, tifffile as tiff, geopandas as gpd
from shapely.geometry import Polygon
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import json, os, math
from tqdm.auto import tqdm

# --- load model + thresholds -------------------------------------------
model = StarDist2D(None, name=MODEL_NAME, basedir=MODEL_BASE)
thr_path = os.path.join(MODEL_BASE, MODEL_NAME, "thresholds.json")
if os.path.exists(thr_path):
    with open(thr_path) as f:
        thr = json.load(f)
    P, N = thr["prob"], thr["nms"]
else:                                   # fallback defaults
    P, N = 0.5, 0.3
print(f"prob_thresh={P:.3f}  nms_thresh={N:.3f}")

# --- read image ---------------------------------------------------------
with rasterio.open(IMG_PATH) as src:
    img_raw = src.read(1).astype("float32")          # H×W
    img     = normalize(img_raw, 0, 1)[..., None]    # H×W×1

    # --- predict instances (StarDist handles tiling internally) ---------
    labels, det = model.predict_instances(
        img,
        axes="YXC",
        prob_thresh=P,
        nms_thresh=N,
        # min_overlap=32,          # 64-px overlap; adapt if crowns are huge
        # block_size=384,          # internal tile size; omit to use default 128
        show_tile_progress=True
    )

    # --- save label image ----------------------------------------------
    profile = src.profile.copy()
    profile.update(
        dtype="uint16",
        count=1,
        compress="lzw",
        nodata=0,  # legal for uint16
    )

    with rasterio.open(OUT_LABELS, "w", **profile) as dst:
        dst.write(labels.astype("uint16"), 1)
    print(f"✓ wrote label raster → {OUT_LABELS}")

    # --- vectorise ------------------------------------------------------
    def poly_from_coord(coords):
        xs, ys = rasterio.transform.xy(src.transform,
                                       coords[:, 0], coords[:, 1],
                                       offset="center")
        if len(coords) < 4:
            return None
        p = Polygon(zip(xs, ys))
        return p if p.is_valid and not p.is_empty else None

    polys = []
    for coord in tqdm(det["coord"], desc="Vectorising"):
        p = poly_from_coord(coord)
        if p:
            polys.append(p)

    gpd.GeoDataFrame(geometry=polys, crs=src.crs)\
       .to_file(OUT_VECT, driver="GPKG")
    print(f"✓ wrote polygons → {OUT_VECT}  ({len(polys)} crowns)")