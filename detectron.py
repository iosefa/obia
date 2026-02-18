#!/usr/bin/env python3
"""
Train a class-agnostic Mask R-CNN (R50-FPN) crown segmenter on WV3 8-band images
using crown polygons from matching GeoPackages.

- Images:  /Users/iosefa/repos/sam/IoU/setA_plot<i>.tif
- Masks:   /Users/iosefa/repos/sam/IoU/validation<i>.gpkg   (others ignored)
- Bands:   NIR, R, G -> indices 7,5,3 (1-indexed in rasterio)
- Output:  /Users/iosefa/repos/sam/maskrcnn_crowns_model_final.pth
           /Users/iosefa/repos/sam/maskrcnn_crowns_config.yaml

Requires: detectron2, rasterio, geopandas, shapely, numpy
"""
import copy
import os
import glob
import json
import random
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid

import torch

# ---- Detectron2 setup ----
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_setup, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.data import build_detection_train_loader, build_detection_test_loader

# -------------------------------
# PATHS / PARAMS (edit if needed)
# -------------------------------
ROOT_DIR = "/Users/iosefa/repos/sam/IoU/"
IMG_GLOB = "setA_plot*.tif"
MASK_GLOB = "validation*.gpkg"   # only these gpkg files are used

# Use WV3 bands: 7 (NIR), 5 (Red), 3 (Green) -- rasterio is 1-indexed
BANDS_1INDEX = [7, 5, 3]

# Split strategy: use all for training by default (since you’ll validate elsewhere).
# If you want a small internal val, set HOLDOUT_FRACTION to e.g. 0.15.
HOLDOUT_FRACTION = 0.0

# Output
OUT_DIR = Path("/Users/iosefa/repos/sam")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_WEIGHTS = OUT_DIR / "maskrcnn_crowns_model_final.pth"
FINAL_CFG_YAML = OUT_DIR / "maskrcnn_crowns_config.yaml"

# Training hyperparams
SEED = 42
IMS_PER_BATCH = 2          # adjust per GPU memory
BASE_LR = 0.00025
MAX_ITERS = 5000           # scale with your data; increase if needed
WARMUP_ITERS = 500
BATCH_SIZE_PER_IMAGE = 128
NUM_WORKERS = 2            # set 0 on MPS if dataloader issues occur
SCORE_THRESH_TEST = 0.5    # used at inference time for predictor


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_3band_wv3(path, bands_1idx=BANDS_1INDEX, to_uint8=True):
    """
    Read selected bands (1-indexed) from an 8-band WV3 GeoTIFF.
    Returns HxWx3 uint8 (0..255) for detectron2 (OpenCV-like) consumption.
    Uses robust per-band normalization (2-98 percentile).
    """
    with rasterio.open(path) as src:
        arr = src.read(bands_1idx)  # shape (3, H, W)
        arr = arr.astype(np.float32)
        # robust per-band normalization
        out = []
        for c in range(arr.shape[0]):
            band = arr[c]
            finite = np.isfinite(band)
            if np.any(finite):
                lo, hi = np.percentile(band[finite], [2, 98])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo, hi = 0.0, 1.0
                band = np.clip(band, lo, hi)
                band = (band - lo) / max(hi - lo, 1e-6)
            else:
                band = np.zeros_like(band, dtype=np.float32)
            out.append(band)
        img = np.stack(out, axis=0)  # (3,H,W)
        img = np.moveaxis(img, 0, -1)  # (H,W,3)
        if to_uint8:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img


def _geom_to_img_coords(geom, src: rasterio.io.DatasetReader):
    """
    Convert a Shapely geometry from CRS of src (georeferenced) to image pixel coordinates.
    """
    # Affine transform from pixel coords to georeferenced: src.transform
    # We need inverse: world -> pixel
    inv_affine = ~src.transform

    def _affine_xy(x, y, z=None):
        col, row = inv_affine * (x, y)
        return (col, row)

    g = make_valid(geom)
    if g.is_empty:
        return None
    # Ensure polygonal
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        g = g.buffer(0)
    if g.is_empty:
        return None
    g_img = shp_transform(_affine_xy, g)
    return g_img


def _poly_to_coco_segmentation(poly: Polygon):
    """
    Convert a single Polygon in image pixel coords to COCO-style segmentation (list of coords).
    """
    if not isinstance(poly, Polygon):
        return []
    exterior = np.array(poly.exterior.coords, dtype=np.float32)
    # Filter very small rings
    if exterior.shape[0] < 3:
        return []
    seg = exterior.flatten().tolist()
    return seg


def build_records(root_dir):
    """
    Build list of dataset dicts for Detectron2 from image+gpkg pairs.
    Only uses gpkg files that match 'validation<i>.gpkg'.
    """
    root = Path(root_dir)
    img_paths = sorted(root.glob(IMG_GLOB))

    # Build a map from index i to gpkg path (validation<i>.gpkg)
    gpkg_paths = sorted(root.glob(MASK_GLOB))
    idx_from_name = lambda p: "".join([ch for ch in p.stem if ch.isdigit()])  # extract digits
    gpkg_by_idx = {idx_from_name(p): p for p in gpkg_paths}

    dataset_dicts = []
    for img_p in img_paths:
        # Find matching gpkg by index
        idx = idx_from_name(img_p)
        if idx not in gpkg_by_idx:
            warnings.warn(f"No matching GPKG for image {img_p.name}; skipping.")
            continue
        gpkg_p = gpkg_by_idx[idx]

        with rasterio.open(img_p) as src:
            H, W = src.height, src.width
            img_crs = src.crs

        # load polygons, reproject to image CRS if needed
        gdf = gpd.read_file(gpkg_p)
        if gdf.empty:
            warnings.warn(f"{gpkg_p.name} contains no polygons; skipping.")
            continue

        if gdf.crs is None:
            warnings.warn(f"{gpkg_p.name} has no CRS; assuming same as image.")
            gdf = gdf.set_crs(img_crs)
        elif img_crs and gdf.crs != img_crs:
            gdf = gdf.to_crs(img_crs)

        # convert geometries to image pixel coords
        annos = []
        with rasterio.open(img_p) as src:
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                g_img = _geom_to_img_coords(geom, src)
                if g_img is None or g_img.is_empty:
                    continue

                # Handle MultiPolygon
                polys = []
                if isinstance(g_img, Polygon):
                    polys = [g_img]
                elif isinstance(g_img, MultiPolygon):
                    polys = [p for p in g_img.geoms if p.area > 0]

                # Build COCO-style segmentations and a tight bbox
                segs = []
                bboxes = []
                for p in polys:
                    seg = _poly_to_coco_segmentation(p)
                    if len(seg) < 6:
                        continue
                    segs.append(seg)
                    x, y, maxx, maxy = p.bounds
                    bboxes.append([x, y, maxx - x, maxy - y])

                if not segs:
                    continue

                # Merge bboxes (one annotation with multi-seg)
                # You can also create one anno per polygon; here we keep one anno per crown geometry
                x0 = min(b[0] for b in bboxes)
                y0 = min(b[1] for b in bboxes)
                x1 = max(b[0] + b[2] for b in bboxes)
                y1 = max(b[1] + b[3] for b in bboxes)
                bbox = [x0, y0, x1 - x0, y1 - y0]

                ann = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": segs,
                    "category_id": 0,   # class-agnostic: single category "crown"
                    "iscrowd": 0,
                }
                annos.append(ann)

        record = {
            "file_name": str(img_p),
            "image_id": img_p.stem,
            "height": H,
            "width": W,
            "annotations": annos,
        }
        dataset_dicts.append(record)

    return dataset_dicts


def register_datasets():
    """
    Register 'crowns_train' and (optional) 'crowns_val' datasets to Detectron2.
    We keep a single category 'crown'.
    """
    all_recs = build_records(ROOT_DIR)
    if not all_recs:
        raise RuntimeError("No training records found. Check paths/patterns.")

    # Optional internal holdout
    if HOLDOUT_FRACTION > 0.0:
        n = len(all_recs)
        k = int(round(n * (1.0 - HOLDOUT_FRACTION)))
        random.shuffle(all_recs)
        train_recs = all_recs[:k]
        val_recs = all_recs[k:]
    else:
        train_recs = all_recs
        val_recs = []

    def get_train():
        return train_recs

    def get_val():
        return val_recs

    DatasetCatalog.register("crowns_train", get_train)
    MetadataCatalog.get("crowns_train").set(thing_classes=["crown"])

    if val_recs:
        DatasetCatalog.register("crowns_val", get_val)
        MetadataCatalog.get("crowns_val").set(thing_classes=["crown"])
    else:
        # still create a dummy val pointing to train to avoid trainer complaints
        DatasetCatalog.register("crowns_val", get_train)
        MetadataCatalog.get("crowns_val").set(thing_classes=["crown"])


# Custom mapper to read 3-band WV3 (NIR,R,G) instead of standard RGB files
def crowns_mapper(dataset_dict):
    d = copy.deepcopy(dataset_dict)

    # --- Read 8-band WV3 with rasterio and pick NIR,R,G (1-indexed 7,5,3) ---
    with rasterio.open(d["file_name"]) as src:
        # read returns (C,H,W) for given indexes
        nir = src.read(7).astype(np.float32)
        red = src.read(5).astype(np.float32)
        grn = src.read(3).astype(np.float32)

    # robust per-band normalization to 0-255 (2–98 percentile)
    def norm255(band):
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

    nir8 = norm255(nir)
    red8 = norm255(red)
    grn8 = norm255(grn)

    # Feed model **RGB** = (R,G,NIR) so cfg.INPUT.FORMAT = "RGB"
    rgb = np.dstack([red8, grn8, nir8])   # (H,W,3), uint8
    d["image"] = torch.as_tensor(rgb.transpose(2, 0, 1))  # (3,H,W), uint8

    # --- Build Instances from annotations (unchanged from your last fix) ---
    annos = [a for a in d.get("annotations", []) if a.get("iscrowd", 0) == 0]
    for a in annos:
        a.setdefault("bbox_mode", BoxMode.XYXY_ABS)
    instances = utils.annotations_to_instances(
        annos, image_size=rgb.shape[:2], mask_format="polygon"  # or "bitmask" if you rasterized
    )
    if instances.has("gt_boxes"):
        instances = instances[instances.gt_boxes.nonempty()]
    d["instances"] = instances
    d.pop("annotations", None)
    return d


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=crowns_mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=crowns_mapper)


def main():
    set_seed(SEED)
    register_datasets()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.INPUT.FORMAT = "RGB"
    cfg.DATASETS.TRAIN = ("crowns_train",)
    cfg.DATASETS.TEST = ("crowns_val",)
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.MAX_ITER = MAX_ITERS
    cfg.SOLVER.STEPS = []  # no LR decay by default; you can add milestones
    cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # class-agnostic: only "crown"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST

    cfg.INPUT.FORMAT = "BGR"  # we fed BGR tensors in mapper
    cfg.OUTPUT_DIR = str(OUT_DIR / "maskrcnn_train_logs")

    cfg.MODEL.DEVICE = "cpu"  # <- required on macOS/MPS
    cfg.SOLVER.IMS_PER_BATCH = 1  # optional: keep memory sane on CPU
    cfg.DATALOADER.NUM_WORKERS = 0

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(FINAL_CFG_YAML, "w") as f:
        f.write(cfg.dump())

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    src_final = Path(cfg.OUTPUT_DIR) / "model_final.pth"
    if src_final.exists():
        os.replace(src_final, FINAL_WEIGHTS)
    else:
        last = Path(cfg.OUTPUT_DIR) / "model_0004999.pth"
        if last.exists():
            os.replace(last, FINAL_WEIGHTS)

    print(f"\nSaved final weights to: {FINAL_WEIGHTS}")
    print(f"Saved config to:        {FINAL_CFG_YAML}")


if __name__ == "__main__":
    main()