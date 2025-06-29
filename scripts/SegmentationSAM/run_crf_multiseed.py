#!/usr/bin/env python3
"""
run_crf_tile.py  –  multi‑seed CRF (seed band IS a probability)
────────────────────────────────────────────────────────────────
• seed_*.tif band‑1  = P(pixel ∈ crown_k)  in [0,1]
• background prior   = min( 0.05 , (CHM₀−chm)/CHM₀ , 0.4 )
• unary(seed k)      = −log( (p_k+ε)/(p_bg+ε) )
• unary(background)  = 0
• pairwise weight    = λ · boundary_cost   (Potts)
"""

import argparse, sys, warnings, importlib
from pathlib import Path
import numpy as np, rasterio as rio, rasterio.warp
from skimage.measure import label as cc_label

# pygco
if not hasattr(np, "float128"): np.float128 = np.longdouble
gco = importlib.import_module("gco"); sys.modules.setdefault("pygco", gco)
from gco import cut_general_graph

SAFE = 30_000
INF  = 30_001
EPS  = 1e-6
CHM_GROUND = 3.0      # m
P_BG_MIN   = 0.05     # non‑zero background everywhere

# ── helpers ───────────────────────────────────────────────────────
def resample_cost(src_path, profile, H, W):
    with rio.open(src_path) as src:
        arr = src.read(1, masked=True).astype(np.float32)
        dst = np.empty((1, H, W), np.float32)
        rio.warp.reproject(arr.filled(0)[None,...], dst,
                           src_transform=src.transform, src_crs=src.crs,
                           dst_transform=profile["transform"],
                           dst_crs=profile["crs"],
                           resampling=rio.warp.Resampling.bilinear)
    cost = dst[0]
    nod  = np.isnan(cost); cost[nod] = 0
    return cost, nod

def build_edges(cost, lam, nod):
    h,w = cost.shape
    wh = (lam*cost[:, :-1]).ravel()
    wv = (lam*cost[:-1, :]).ravel()
    wh[(nod[:, :-1]|nod[:, 1:]).ravel()] = 0
    wv[(nod[:-1, :]|nod[1:, :]).ravel()] = 0
    wh = np.clip(wh,0,SAFE).astype(np.int32)
    wv = np.clip(wv,0,SAFE).astype(np.int32)
    idx = np.arange(h*w,dtype=np.int32).reshape(h,w)
    edges = np.column_stack([np.concatenate([idx[:, :-1].ravel(),
                                             idx[:-1, :].ravel()]),
                             np.concatenate([idx[:,  1:].ravel(),
                                             idx[1:,  :].ravel()])])
    weights = np.concatenate([wh,wv])
    return edges, weights

def anchor_pixel(prob, ground, nod):
    """highest‑prob canopy, otherwise global max"""
    order = prob.ravel().argsort()[::-1]
    for idx in order:
        y,x = divmod(idx, prob.shape[1])
        if ground[y,x] < .5 and not nod[y,x]:
            return y,x
    return np.unravel_index(prob.argmax(), prob.shape)

# ── main ──────────────────────────────────────────────────────────
def crf_tile(tile_dir: Path, cost_raster: Path, a):
    seeds = sorted(tile_dir.glob("seed_*.tif"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not seeds:
        raise SystemExit(f"{tile_dir}: no seed_*.tif found")

    with rio.open(seeds[0]) as ds0:
        ground = ds0.read(2).astype(np.float32)         # 1 ground
        chm    = ds0.read(3).astype(np.float32)
        prof   = ds0.profile
        H,W    = prof["height"], prof["width"]

    cost,nod = resample_cost(cost_raster, prof, H, W)

    probs=[]; anchors=[]
    for f in seeds:
        with rio.open(f) as ds:
            p = ds.read(1).astype(np.float32)           # already probability
        p_max = p.max()
        if p_max > 0:
            p *= 0.9 / p_max                            # normalise
        probs.append(p)
        anchors.append(anchor_pixel(p, ground, nod))
    probs = np.stack(probs)                             # (N,H,W)
    N     = probs.shape[0]

    # background probability (0.05…0.4)
    p_bg = np.clip((CHM_GROUND - chm)/CHM_GROUND, 0, 0.4)
    p_bg = np.maximum(p_bg, P_BG_MIN)

    # unary energies (odds)
    unaries = np.empty((N+1, H, W), np.float32)
    unaries[0]  = 0
    unaries[1:] = -np.log((probs+EPS)/(p_bg+EPS))
    unary_i = np.clip(unaries * a.scale, -SAFE, SAFE).astype(np.int32)

    # hard anchor pixel
    for lab,(y,x) in enumerate(anchors,1):
        unary_i[:,y,x]   =  INF
        unary_i[lab,y,x] = -INF

    unary = unary_i.transpose(1,2,0).reshape(-1, N+1)

    edges,weights = build_edges(cost, a.lam, nod)
    pair = np.ones((N+1,N+1),np.int32) - np.eye(N+1,dtype=np.int32)

    labels = cut_general_graph(edges,weights,unary,pair,
                               n_iter=a.iter, algorithm="expansion")\
             .reshape(H,W).astype(np.int32)
    labels[nod | (ground>.5)] = -1

    # restrict to component containing anchor
    for lab,(y,x) in enumerate(anchors,1):
        labels[y,x] = lab
        comp = cc_label(labels==lab, connectivity=2)
        keep = comp[y,x]
        labels[(labels==lab)&(comp!=keep)] = 0

    if a.verbose:
        crowns=np.unique(labels[labels>0]).size
        print(f"{tile_dir.name}: seeds={N}  crowns={crowns}")

    prof.update(count=1,dtype="int32",nodata=-1,compress="deflate")
    with rio.open(tile_dir/'labels.tif','w',**prof) as dst:
        dst.write(labels,1)

# ── CLI ───────────────────────────────────────────────────────────
if __name__=="__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("tile_dir"); pa.add_argument("cost")
    pa.add_argument("--scale", type=float, default=400,
                    help="unary × scale (default 400)")
    pa.add_argument("--lam",   type=float, default=5,
                    help="pairwise λ (default 5)")
    pa.add_argument("--iter",  type=int,   default=5,
                    help="α‑expansion iterations")
    pa.add_argument("--verbose", action="store_true")
    args = pa.parse_args()

    try:
        crf_tile(Path(args.tile_dir), Path(args.cost), args)
    except Exception as e:
        warnings.warn(str(e)); sys.exit(1)

# #!/usr/bin/env python3
# """
# Pixel-level multi-seed CRF with a CHM-based *background* label
# -------------------------------------------------------------
# Each canonical-seed band + one background band compete on the full
# 4-neighbour pixel graph.
#
# Inputs
# ------
# logits.tif : bands 1…N  centred SAM logits  (one band per canonical seed)
#              band N+1   ground mask   (1 = ground, 0 = canopy)
#              band N+2   CHM resampled (metres)
# cost.tif   : boundary-cost 0-1  (any resolution / nodata OK)
#
# Outputs
# -------
# labels.tif : -1 nodata/ground • 0 background • 1…N crowns
# """
# import numpy as np, rasterio as rio, rasterio.warp, argparse, sys, importlib
# if not hasattr(np, "float128"): np.float128 = np.longdouble
# gco = importlib.import_module("gco"); sys.modules.setdefault("pygco", gco)
# from gco import cut_general_graph
#
# SAFE = 30_000                       # upper edge-weight clip (fits int16)
# CHM_GROUND = 3.0                    # m – canopy / ground threshold
#
#
# # ───────── small helpers ────────────────────────────────────────────
# def resample_cost(path, dst_profile, H, W):
#     """Bilinear-resample cost raster to logits grid; return array & mask."""
#     with rio.open(path) as src:
#         src_arr = src.read(1, masked=True).astype(np.float32)
#         dst_arr = np.empty((1, H, W), np.float32)
#         rio.warp.reproject(
#             src_arr.filled(0)[None, ...], dst_arr,
#             src_transform=src.transform, src_crs=src.crs,
#             dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
#             resampling=rio.warp.Resampling.bilinear)
#     cost = dst_arr[0]
#     nod = np.isnan(cost)
#     cost[nod] = 0.0
#     return cost, nod
#
#
# def build_edges(cost, lam, escale, nod):
#     """Return 4-neighbour edge list + int32 weights (Potts)."""
#     h, w = cost.shape
#     wh = (lam * cost[:, :-1] * escale).ravel()
#     wv = (lam * cost[:-1, :] * escale).ravel()
#
#     mask_h = nod[:, :-1] | nod[:, 1:]
#     mask_v = nod[:-1, :] | nod[1:, :]
#     wh[mask_h.ravel()] = 0
#     wv[mask_v.ravel()] = 0
#
#     wh = np.clip(wh, 0, SAFE).astype(np.int32)
#     wv = np.clip(wv, 0, SAFE).astype(np.int32)
#
#     idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
#     edges = np.column_stack([
#         np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()]),
#         np.concatenate([idx[:,  1:].ravel(), idx[1:,  :].ravel()])
#     ])
#     weights = np.concatenate([wh, wv])
#     return edges, weights
# # -------------------------------------------------------------------
#
#
# def main(a):
#     # 1 ─ Read logits stack -----------------------------------------
#     with rio.open(a.logits) as ds:
#         stack = ds.read().astype(np.float32)
#         prof  = ds.profile
#         H, W  = prof["height"], prof["width"]
#
#     seed_logits = stack[:-2]                   # (N,H,W)
#     mask        = stack[-2]                   # ground mask
#     chm         = stack[-1]
#     N           = seed_logits.shape[0]
#
#     # 2 ─ Build *background* logit from CHM -------------------------
#     #   low CHM ⇒ very positive bg-logit (favors background)
#     #   canopy   ⇒ negative bg-logit   (lets seed labels win)
#     p_bg_soft = np.clip((CHM_GROUND - chm) / CHM_GROUND, 0, 1)   # 1…0
#     bg_logit  = np.log(p_bg_soft + 1e-6) - np.log(1 - p_bg_soft + 1e-6)
#
#     # 3 ─ Combine & soft-max (probabilities sum to 1) ---------------
#     all_logits = np.concatenate([bg_logit[None, ...], seed_logits], 0)   # (N+1,H,W)
#     max_per_px = all_logits.max(0, keepdims=True)
#     expx = np.exp(all_logits - max_per_px)
#     probs = expx / expx.sum(0, keepdims=True)                            # (N+1,H,W)
#
#     # 4 ─ Unary energies  (−log P × scale) --------------------------
#     unary_f = -np.log(probs + 1e-9)
#     unary_i = (unary_f * a.scale).clip(0, SAFE).astype(np.int32)
#
#     if a.label_cost:
#         penalty = np.zeros(N + 1, np.int32)
#         penalty[1:] = a.label_cost
#         unary_i += penalty[:, None, None]
#
#     unary = unary_i.transpose(1, 2, 0).reshape(-1, N + 1)
#
#     # 5 ─ Boundary cost & edge list --------------------------------
#     cost, nod_px = resample_cost(a.cost, prof, H, W)
#     medU, medC = np.median(unary_f), np.median(cost[~nod_px])
#     escale = max(1.0, min(SAFE / a.lam, medU / (medC + 1e-9)))
#     edges, weights = build_edges(cost, a.lam, escale, nod_px)
#
#     # 6 ─ α-expansion (Potts) ---------------------------------------
#     pair = np.ones((N + 1, N + 1), np.int32) - np.eye(N + 1, dtype=np.int32)
#     labels = cut_general_graph(
#         edges, weights, unary, pair,
#         n_iter=a.iter, algorithm="expansion"
#     ).reshape(H, W).astype(np.int32)
#
#     labels[nod_px | (mask > 0.5)] = -1        # force nodata/ground
#
#     if a.verbose:
#         print(f"seeds {N}  edge-scale {escale:.1f}")
#
#     # 7 ─ Write result ---------------------------------------------
#     prof.update(count=1, dtype="int32", nodata=-1, compress="deflate")
#     with rio.open(a.out, "w", **prof) as dst:
#         dst.write(labels, 1)
#
#     n_crowns = np.unique(labels[labels > 0]).size
#     pix      = np.count_nonzero(labels > 0)
#     print(f"✓ crowns: {n_crowns:,}  pixels: {pix:,}")
#
#
# # ───────── CLI ──────────────────────────────────────────────────────
# if __name__ == "__main__":
#     pa = argparse.ArgumentParser(description="Pixel-level multi-seed CRF with CHM background")
#     pa.add_argument("logits"); pa.add_argument("cost"); pa.add_argument("out")
#     pa.add_argument("--scale", type=float, default=1000,
#                     help="× for −log P → int32 (default 1000)")
#     pa.add_argument("--lam",   type=float, default=25,
#                     help="edge weight λ (default 25)")
#     pa.add_argument("--label-cost", type=int, default=200,
#                     help="per-seed penalty (0 = off)")
#     pa.add_argument("--iter",  type=int, default=5,
#                     help="α-expansion iterations (default 5)")
#     pa.add_argument("--verbose", action="store_true")
#     main(pa.parse_args())

# #!/usr/bin/env python3
# """
# Multi‑seed CRF crown segmentation
#
# Inputs
# ------
# logits.tif : bands 1…N seed logits, band N+1 ground‑mask, band N+2 CHM
# cost.tif   : boundary cost (0–1)
#
# Outputs
# -------
# labels.tif : -1 nodata/ground • 0 background • 1…N crowns
# """
# import numpy as np, rasterio as rio, rasterio.warp, argparse, importlib, sys
# if not hasattr(np, "float128"): np.float128 = np.longdouble
# gco = importlib.import_module("gco"); sys.modules.setdefault("pygco", gco)
# from gco import cut_general_graph
# SAFE = 30000
#
#
# def softmax(logits):
#     xm = logits.max(axis=0, keepdims=True)
#     ex = np.exp(logits - xm)
#     return ex / ex.sum(axis=0, keepdims=True)
#
#
# def resample_cost(cost_path, dst_profile, H, W):
#     with rio.open(cost_path) as src:
#         arr = src.read(1, masked=True).astype(np.float32)
#         src_arr = arr.filled(0)[np.newaxis, ...]
#         dst_arr = np.empty((1, H, W), np.float32)
#         rasterio.warp.reproject(
#             src_arr, dst_arr,
#             src_transform=src.transform, src_crs=src.crs,
#             dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
#             resampling=rasterio.warp.Resampling.bilinear)
#     cost = dst_arr[0]
#     nodata = np.isnan(cost)
#     cost[nodata] = 0.0
#     return cost, nodata
#
#
# def build_edges(cost, lam, medU, medC, nodata):
#     scale = min(SAFE/lam, medU/ (medC+1e-9))
#     h, w = cost.shape
#     wh = lam * cost[:, :-1] * scale
#     wv = lam * cost[:-1, :] * scale
#     mH = nodata[:, :-1] | nodata[:, 1:]
#     mV = nodata[:-1, :] | nodata[1:, :]
#     wh[mH] = wv[mV] = 0
#     wh = np.clip(wh, 0, SAFE).astype(np.int32).ravel()
#     wv = np.clip(wv, 0, SAFE).astype(np.int32).ravel()
#     idx = np.arange(h*w, dtype=np.int32).reshape(h, w)
#     edges = np.column_stack([np.concatenate([idx[:, :-1].ravel(),
#                                              idx[:-1,:].ravel()]),
#                              np.concatenate([idx[:, 1:].ravel(),
#                                              idx[1:, :].ravel()])])
#     return edges, np.concatenate([wh, wv]), scale
#
#
# def main(a):
#     # ---- read logits ------------------------------------------------
#     with rio.open(a.logits) as ds:
#         arr = ds.read().astype(np.float32)
#         prof = ds.profile
#     *seed_logits, mask, chm = arr
#     N, H, W = len(seed_logits), *seed_logits[0].shape
#
#     # ---- per‑tile standardisation ----------------------------------
#     canopy = mask < 0.5
#     logits_std = []
#     for log in seed_logits:
#         shift = np.median(log[canopy])
#         mad   = np.median(np.abs(log[canopy]-shift))+1e-6
#         logits_std.append((log-shift)/mad)
#     logits_std = np.stack(logits_std, 0)
#
#     # ---- probabilities --------------------------------------------
#     p_seed = softmax(logits_std)                        # (N,H,W)
#     bg_soft = np.clip((3.0 - chm) / 3.0, 0, 1)          # 0..1
#     p_bg = np.maximum(bg_soft, 1 - p_seed.max(axis=0))
#     probs = np.concatenate([p_bg[None,...], p_seed], 0) # (N+1,H,W)
#
#     # ---- unary  −log P × scale ------------------------------------
#     unary_f = -np.log(probs + 1e-12)
#     unary_i = np.nan_to_num(unary_f * a.scale, nan=SAFE, posinf=SAFE).astype(np.int32)
#     if a.label_cost:
#         penalty = np.zeros(N+1, np.int32); penalty[1:] = a.label_cost
#         unary_i += penalty[:,None,None]
#     unary = unary_i.transpose(1,2,0).reshape(-1, N+1)
#
#     # ---- cost raster ----------------------------------------------
#     cost, nodata = resample_cost(a.cost, prof, H, W)
#
#     # ---- edges -----------------------------------------------------
#     edges, weights, escale = build_edges(cost, a.lam,
#                                          np.median(unary_f),
#                                          np.median(cost), nodata)
#
#     # ---- α‑expansion ----------------------------------------------
#     pair = np.ones((N + 1, N + 1), np.int32) - np.eye(N + 1,     dtype=np.int32)
#     labels = cut_general_graph(edges, weights, unary, pair,
#                                n_iter=a.iter, algorithm="expansion") \
#              .reshape(H,W).astype(np.int32)
#     labels[nodata | (mask>0.5)] = -1
#
#     if a.verbose:
#         print(f"seeds {N}  edge‑scale {escale:.1f}")
#
#     # ---- write -----------------------------------------------------
#     prof.update(count=1, dtype="int32", nodata=-1, compress="deflate")
#     with rio.open(a.out, "w", **prof) as dst:
#         dst.write(labels, 1)
#     print(f"✓ crowns: {np.unique(labels[labels>0]).size:,}  "
#           f"pixels: {np.count_nonzero(labels>0):,}")
#
#
# if __name__ == "__main__":
#     P = argparse.ArgumentParser(description="Multi‑seed CRF crowns")
#     P.add_argument("logits"); P.add_argument("cost"); P.add_argument("out")
#     P.add_argument("--scale", type=float, default=800)
#     P.add_argument("--lam", type=float, default=15)
#     P.add_argument("--label-cost", type=int, default=300)
#     P.add_argument("--iter", type=int, default=5)
#     P.add_argument("--verbose", action="store_true")
#     main(P.parse_args())