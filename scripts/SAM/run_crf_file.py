#!/usr/bin/env python3
"""
run_crf_file.py  – cost-aware CRF per *individual* seed_XXX.tif

Band convention inside each seed file
  1  SAM probability (float32 0-1)     –- required
  2  ground mask  (float32 0/1)        –- optional
  3  CHM          (float32 metres)     –- optional

If bands 2–3 are absent they are warped in from the global CHM raster.

Outputs
  labels_<n>.tif   (same size / geotransform as the seed file)
  └─ int32, nodata = –1,  0 = background,  1 = crown

One can merge all label patches afterwards with the helper function
`mosaic_labels()` that is included at the bottom of this script.
"""
# ------------------------------------------------------------------
import argparse, sys, warnings, importlib
from pathlib import Path

import numpy as np, rasterio as rio, rasterio.warp
from skimage.measure import label as cc_label

# ---- pygco shim (works on Apple silicon) -------------------------
if not hasattr(np, "float128"):
    np.float128 = np.longdouble
gco = importlib.import_module("gco"); sys.modules.setdefault("pygco", gco)
from gco import cut_general_graph
# ------------------------------------------------------------------

SAFE, INF, EPS           = 30_000, 30_001, 1e-6
CHM_GROUND, P_BG_MIN     = 3.0, 0.05      # background prior params

# ────────── helper functions ──────────────────────────────────────
def warp_to_seed(src_ds: rio.io.DatasetReader, seed_prof: dict) -> np.ndarray:
    """Warp *a single-band* dataset into the seed file’s grid."""
    arr = src_ds.read(1, masked=True).astype(np.float32)
    dst = np.empty((1, seed_prof["height"], seed_prof["width"]), np.float32)
    rio.warp.reproject(
        arr.filled(np.nan)[None, ...], dst,
        src_transform=src_ds.transform, src_crs=src_ds.crs,
        dst_transform=seed_prof["transform"], dst_crs=seed_prof["crs"],
        resampling=rio.warp.Resampling.bilinear)
    return dst[0]

def crop_and_warp(path: Path, seed_prof: dict) -> np.ndarray:
    with rio.open(path) as ds:
        return warp_to_seed(ds, seed_prof)

def cost_patch(cost_path: Path, seed_prof: dict):
    with rio.open(cost_path) as src:
        patch = warp_to_seed(src, seed_prof)
    nod = np.isnan(patch)
    patch[nod] = 0.0
    return patch.astype(np.float32), nod

def build_edges(cost, lam, nod):
    h, w = cost.shape
    wh = np.clip(lam * cost[:, :-1], 0, SAFE).ravel()
    wv = np.clip(lam * cost[:-1, :], 0, SAFE).ravel()
    wh[(nod[:, :-1] | nod[:, 1:]).ravel()] = 0
    wv[(nod[:-1, :] | nod[1:, :]).ravel()] = 0
    idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
    edges = np.column_stack([
        np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()]),
        np.concatenate([idx[:,  1:].ravel(), idx[1:,  :].ravel()])])
    weights = np.concatenate([wh, wv]).astype(np.int32)
    return edges, weights

def anchor_pixel(prob, ground, nod):
    """Return the pixel (row, col) to hard-anchor this seed."""
    for idx in prob.ravel().argsort()[::-1]:
        y, x = divmod(idx, prob.shape[1])
        if ground[y, x] < .5 and not nod[y, x]:
            return y, x
    return np.unravel_index(prob.argmax(), prob.shape)

# ────────── CRF for ONE seed file ─────────────────────────────────
def crf_seed(seed_file: Path, cost_path: Path, chm_path: Path,
             scale: float, lam: float, iters: int, verbose=False):

    sid = seed_file.stem.split("_")[1]

    with rio.open(seed_file) as ds:
        prof_seed = ds.profile
        H, W      = prof_seed["height"], prof_seed["width"]

        prob = ds.read(1).astype(np.float32)
        has_ground = ds.count >= 2
        has_chm    = ds.count >= 3
        if has_ground:
            ground = ds.read(2).astype(np.float32)
        if has_chm:
            chm = ds.read(3).astype(np.float32)

    # resample cost, CHM, ground to seed grid if needed
    cost, nod = cost_patch(cost_path, prof_seed)

    if not has_chm or not has_ground:
        chm_global = crop_and_warp(chm_path, prof_seed)
        if not has_chm:
            chm = chm_global
        if not has_ground:
            ground = (chm_global < CHM_GROUND).astype(np.float32)

    # normalise probability so max == 0.9 (prevents −log overflow)
    m = prob.max()
    if m > 0:
        prob *= 0.9 / m

    # background prior
    p_bg = np.clip((CHM_GROUND - chm) / CHM_GROUND, 0, 0.4)
    p_bg = np.maximum(p_bg, P_BG_MIN)

    # unaries (two labels: 0 = bg, 1 = this crown)
    un   = np.empty((2, H, W), np.float32)
    un[0] = 0.0
    un[1] = -np.log((prob + EPS) / (p_bg + EPS))
    un_i  = np.clip(un * scale, -SAFE, SAFE).astype(np.int32)

    y0, x0 = anchor_pixel(prob, ground, nod)
    un_i[:, y0, x0]   =  INF
    un_i[1, y0, x0]   = -INF
    unary = un_i.transpose(1, 2, 0).reshape(-1, 2)

    edges, weights = build_edges(cost, lam, nod)
    potts = np.array([[0, 1], [1, 0]], np.int32)

    labels = cut_general_graph(edges, weights, unary, potts,
                               n_iter=iters, algorithm="expansion")\
             .reshape(H, W).astype(np.int32)

    labels[nod | (ground > .5)] = -1     # nodata / ground → void

    # keep only connected component containing the anchor
    comp = cc_label(labels == 1, connectivity=2)
    keep = comp[y0, x0]
    labels[(labels == 1) & (comp != keep)] = 0

    out_path = seed_file.with_name(f"labels_{sid}.tif")
    prof_seed.update(count=1, dtype="int32", nodata=-1, compress="deflate")
    with rio.open(out_path, "w", **prof_seed) as dst:
        dst.write(labels, 1)

    if verbose:
        print(f"seed {sid}:  crown px = {(labels == 1).sum():,}")
    return out_path


# ────────── convenience: mosaic all label patches  ────────────────
def mosaic_labels(label_files, ref_raster, out_path):
    """Merge all label patches >0 into a single raster using max()."""
    with rio.open(ref_raster) as ref:
        prof = ref.profile
        H, W = prof["height"], prof["width"]
        mosaic = np.full((H, W), -1, np.int32)

    for lf in label_files:
        with rio.open(lf) as ds:
            lab = ds.read(1)
            mask = lab > mosaic
            mosaic[mask] = lab[mask]

    prof.update(dtype="int32", nodata=-1, compress="deflate")
    with rio.open(out_path, "w", **prof) as dst:
        dst.write(mosaic, 1)
    return out_path


# ────────── command-line interface  ───────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="CRF crown segmentation per seed")
    pa.add_argument("seed_dir",  help="folder containing seed_*.tif")
    pa.add_argument("cost_tif",  help="boundary-cost raster (0–1)")
    pa.add_argument("chm_tif",   help="global CHM raster")
    pa.add_argument("--scale", type=float, default=400)
    pa.add_argument("--lam",   type=float, default=5)
    pa.add_argument("--iter",  type=int,   default=5)
    pa.add_argument("--merge", help="optional path for mosaic labels")
    pa.add_argument("--verbose", action="store_true")
    args = pa.parse_args()

    seed_dir = Path(args.seed_dir)
    cost_tif = Path(args.cost_tif)
    chm_tif  = Path(args.chm_tif)

    label_paths = []
    for seed_file in sorted(seed_dir.glob("seed_*.tif")):
        try:
            lp = crf_seed(seed_file, cost_tif, chm_tif,
                          scale=args.scale, lam=args.lam,
                          iters=args.iter, verbose=args.verbose)
            label_paths.append(lp)
        except Exception as exc:
            warnings.warn(f"{seed_file.name}: {exc}")

    if args.merge:
        mosaic = mosaic_labels(label_paths, cost_tif, Path(args.merge))
        print(f"✓ mosaic → {mosaic}")