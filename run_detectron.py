#!/usr/bin/env python3
import os
from pathlib import Path
import json
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import features
from affine import Affine
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

import torch
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.engine import DefaultPredictor

# -----------------------------
# USER INPUTS
# -----------------------------
MODEL_PTH   = "/Users/iosefa/repos/sam/maskrcnn_crowns_model_final.pth"  # your trained weights
INPUT_TIF   = "/Users/iosefa/repos/sam/area51_subset1.tif"            # target area GeoTIFF
OUTPUT_GPKG = "/Users/iosefa/repos/sam/crowns_predicted.gpkg"
OUTPUT_LAYER= "crowns_pred"

# Bands to use (1-indexed WV3): NIR,R,G = 7,5,3
BANDS_753   = (7, 5, 3)

# Sliding window
TILE_SIZE   = 1024        # pixels per side
TILE_OVERLAP= 128         # pixels overlap on each side
SCORE_THRESH= 0.5         # drop low-confidence instances
MIN_MASK_PX = 50          # drop tiny masks (in pixels)
NMS_IOU     = 0.5         # IoU threshold for cross-tile dedup (on boxes)

# Device: "cpu" recommended on Mac unless you confirmed CUDA
DEVICE      = "cpu"

# -----------------------------
# HELPERS
# -----------------------------
def robust_norm255(band):
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(band[finite], [2, 98])
    if hi <= lo:
        lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(band, dtype=np.uint8)
    band = np.clip(band, lo, hi)
    band = (band - lo) / max(hi - lo, 1e-6) * 255.0
    return band.astype(np.uint8)

def read_tile_rgb(src, row_off, col_off, size, band_idx=BANDS_753):
    h = min(size, src.height - row_off)
    w = min(size, src.width  - col_off)
    if h <= 0 or w <= 0:
        return None, None
    win = Window(col_off, row_off, w, h)
    # read bands 7,5,3 → (H,W)
    nir = src.read(band_idx[0], window=win).astype(np.float32)
    red = src.read(band_idx[1], window=win).astype(np.float32)
    grn = src.read(band_idx[2], window=win).astype(np.float32)
    nir8, red8, grn8 = robust_norm255(nir), robust_norm255(red), robust_norm255(grn)
    # RGB order (R,G,NIR) to match cfg.INPUT.FORMAT="RGB"
    rgb = np.dstack([red8, grn8, nir8]).astype(np.uint8)  # (H,W,3)
    return rgb, win

def tile_indices(width, height, size, overlap):
    step = size - overlap
    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            yield int(row_off), int(col_off)

def boxes_iou_xyxy(a, b):
    # a: [N,4], b:[M,4]  -> IoU [N,M]
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    union = area_a[:,None] + area_b[None,:] - inter
    return np.where(union > 0, inter/union, 0.0)

def nms_global(boxes, scores, iou_thr=0.5):
    # simple NMS over all boxes
    idxs = np.argsort(-scores)
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = boxes_iou_xyxy(boxes[i:i+1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thr]
    return keep

def mask_to_polygons(mask_bool, transform):
    # mask_bool: (H,W) bool
    if mask_bool.sum() < MIN_MASK_PX:
        return []
    # rasterio.features.shapes expects values; we pass uint8
    for geom, val in features.shapes(mask_bool.astype(np.uint8), mask=mask_bool, transform=transform):
        if val == 0:
            continue
        poly = shape(geom)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue
        yield poly

# -----------------------------
# BUILD PREDICTOR
# -----------------------------
def build_predictor(weights_path, score_thresh=0.5, device=DEVICE):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1          # "crown"
    cfg.INPUT.FORMAT = "RGB"                     # we pass RGB tiles
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device
    cfg.DATALOADER.NUM_WORKERS = 0
    return DefaultPredictor(cfg)

# -----------------------------
# MAIN
# -----------------------------
def main():
    predictor = build_predictor(MODEL_PTH, SCORE_THRESH, DEVICE)

    geoms = []
    scores_all = []
    areas_m2 = []
    tile_ids = []

    with rasterio.open(INPUT_TIF) as src:
        crs = src.crs
        base_transform = src.transform
        res_x = base_transform.a   # pixel size x
        res_y = -base_transform.e  # pixel size y (positive)
        pixel_area = abs(res_x * res_y)

        H, W = src.height, src.width
        print(f"Image: {W} x {H} | CRS={crs}")

        all_boxes = []
        all_scores = []
        all_polys = []

        for t, (row_off, col_off) in enumerate(tile_indices(W, H, TILE_SIZE, TILE_OVERLAP), start=1):
            rgb, win = read_tile_rgb(src, row_off, col_off, TILE_SIZE, BANDS_753)
            if rgb is None:
                continue

            outputs = predictor(rgb)
            instances = outputs["instances"].to("cpu")
            if len(instances) == 0:
                continue

            # tile-local → global pixel coords for boxes
            boxes = instances.pred_boxes.tensor.numpy()  # (N,4) in tile pixels
            boxes[:, [0,2]] += col_off
            boxes[:, [1,3]] += row_off
            scores = instances.scores.numpy()
            masks  = instances.pred_masks.numpy()  # (N,h,w) tile coords

            # Geotransform for this window
            win_transform = src.window_transform(win)

            # convert masks → polygons (geo coords)
            for k in range(masks.shape[0]):
                if scores[k] < SCORE_THRESH:
                    continue
                # polygonize mask_k with window transform
                polys = list(mask_to_polygons(masks[k].astype(bool), win_transform))
                if not polys:
                    continue
                # you may union parts of the same instance
                poly = unary_union(polys)
                if poly.is_empty:
                    continue

                all_polys.append(poly)
                all_boxes.append(boxes[k].copy())
                all_scores.append(float(scores[k]))
                tile_ids.append(t)

        if not all_polys:
            print("No crowns detected above threshold.")
            return

        all_boxes = np.array(all_boxes, dtype=np.float32)
        all_scores = np.array(all_scores, dtype=np.float32)

        # Global NMS on boxes to drop duplicates across overlapping tiles
        keep_idx = nms_global(all_boxes, all_scores, iou_thr=NMS_IOU)
        print(f"Detections: {len(all_polys)} | kept after NMS: {len(keep_idx)}")

        kept_geoms = []
        kept_scores= []
        kept_area  = []
        kept_tile  = []

        for i in keep_idx:
            g = all_polys[i]
            # compute area in m^2 using CRS units (assumes projected CRS)
            area = g.area
            if pixel_area and crs and crs.is_projected:
                # g is already in CRS units; .area is m^2 if CRS meters
                pass
            kept_geoms.append(g)
            kept_scores.append(all_scores[i])
            kept_area.append(area)
            kept_tile.append(tile_ids[i])

        gdf = gpd.GeoDataFrame({
            "score": kept_scores,
            "area_m2": kept_area,
            "tile_id": kept_tile
        }, geometry=kept_geoms, crs=crs)

        # write GeoPackage
        out_path = Path(OUTPUT_GPKG)
        if out_path.exists():
            try: out_path.unlink()
            except Exception: pass
        gdf.to_file(OUTPUT_GPKG, layer=OUTPUT_LAYER, driver="GPKG")
        print(f"Wrote {OUTPUT_GPKG} (layer='{OUTPUT_LAYER}') | crowns={len(gdf)}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()