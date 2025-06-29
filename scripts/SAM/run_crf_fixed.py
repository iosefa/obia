#!/usr/bin/env python
"""
Seed-guided CRF crown segmentation  –  tuned version
====================================================
Inputs
  cost.tif          1-band float32   (edge weights; high at crown borders)
  chm.tif           1-band float32   (optional; sharpens cost map)
  logits_dir/*.tif  1-band float32   (per-seed probability OR logits)
Outputs
  labels.tif                     1-band uint32 (crown IDs; 0 = background)
  crowns/crown_#####.tif         one uint8 mask per crown
"""
import glob, os, sys, argparse, importlib, warnings
import numpy as np, rasterio
from rasterio.enums import Resampling
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

# -- pygco / gco ------------------------------------------------------------
if not hasattr(np, "float128"):
    np.float128 = np.longdouble          # pygco requires this attribute
gco = importlib.import_module("gco"); sys.modules.setdefault("pygco", gco)
from gco import cut_grid_graph

# ---------------------------------------------------------------- CLI ------
P = argparse.ArgumentParser()
P.add_argument("logit_dir")
P.add_argument("cost_tif")
P.add_argument("chm_tif")                 # dummy path allowed if no CHM
P.add_argument("--scale", type=float, default=50,
               help="unary_scale (data-term weight, default 50)")
P.add_argument("--lam",   type=float, default=12,
               help="λ smoothness multiplier (default 12)")
P.add_argument("--iter",  type=int,   default=5,
               help="α-expansion iterations (default 5)")
P.add_argument("--beta",  type=float, default=0.10,
               help="max background prior (default 0.10)")
P.add_argument("--seed_tol", type=float, default=0.02,
               help="drop seeds whose max(prob) ≤ this (default 0.02)")
P.add_argument("--verbose", action="store_true")
args = P.parse_args()

# ------------------------------------------------ helper -------------------
def read_band(path, like=None, res=Resampling.bilinear):
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(1, like.height, like.width),
            resampling=res) if like else src.read(1)
    return data.astype(np.float32)

# -------------------------------- 1. cost raster --------------------------
with rasterio.open(args.cost_tif) as cost_src:
    cost  = cost_src.read(1).astype(np.float32)
    H, W  = cost_src.height, cost_src.width
    prof  = cost_src.profile.copy()

# ------------- 2. (optional) blend CHM gradient into cost -----------------
try:
    chm  = read_band(args.chm_tif, like=cost_src)
    grad = gaussian_gradient_magnitude(chm, sigma=1)
    cost = (cost + grad) * 0.5                    # simple average
except Exception as e:
    warnings.warn(f"CHM gradient skipped: {e}")

# -------------------------------- 3. seed rasters -------------------------
paths = sorted(glob.glob(os.path.join(args.logit_dir, "*.tif")))
if not paths:
    sys.exit(f"No *.tif files in {args.logit_dir}")

probs = np.empty((len(paths), H, W), np.float32)
for i, pth in enumerate(tqdm(paths, desc="loading seeds")):
    probs[i] = read_band(pth, like=cost_src)

# light Gaussian blur so CRF can grow beyond one pixel
probs = np.stack([gaussian_filter(p, sigma=1, truncate=2.0) for p in probs])

# convert logits → probabilities if needed
if probs.min() < 0 or probs.max() > 1:
    probs = 1.0 / (1.0 + np.exp(-probs))

# drop empty seeds
keep  = np.where(probs.max(axis=(1, 2)) > args.seed_tol)[0]
probs = probs[keep]
paths = [paths[i] for i in keep]
n_seeds = len(paths)
if n_seeds == 0:
    sys.exit("All seeds were empty after thresholding")

# ---------------------------- 4. probability cube -------------------------
bg   = np.clip(1.0 - probs.max(axis=0), 1e-6, args.beta)
cube = np.concatenate([bg[None, ...], probs], axis=0)    # (L, H, W)
cube /= cube.sum(axis=0, keepdims=True)                  # per-pixel Σ=1
L = cube.shape[0]

# ---------------------------- 5. unary term -------------------------------
unary = (-np.log(cube) * args.scale).astype(np.int32)    # (L, H, W)
unary = unary.transpose(1, 2, 0).copy(order="C")         # (H, W, L)

# hard-clamp the highest-prob pixel of each seed
for lbl, p in enumerate(cube[1:], start=1):
    y, x = np.unravel_index(np.argmax(p), p.shape)
    unary[y, x, :] = 10_000
    unary[y, x, lbl] = 0

# ---------------------------- 6. pairwise (Potts) -------------------------
pairwise = np.ones((L, L), np.int32)
np.fill_diagonal(pairwise, 0)

# ---------------------------- 7. edge costs -------------------------------
h = args.lam * (cost[:, :-1] + cost[:, 1:]) * 0.5
v = args.lam * (cost[:-1, :] + cost[1:, :]) * 0.5
h = np.clip(h * args.scale, 0, np.iinfo(np.int32).max).astype(np.int32)
v = np.clip(v * args.scale, 0, np.iinfo(np.int32).max).astype(np.int32)

# ---------------------------- 8. optimisation -----------------------------
if args.verbose:
    print(f"Graph: {H}×{W} px, {L} labels (λ={args.lam}, scale={args.scale})")

labels = cut_grid_graph(unary, pairwise, v, h,
                        algorithm="expansion", n_iter=args.iter
                        ).astype(np.uint32).reshape(H, W)

# ---------------------------- 9. write outputs ----------------------------
prof.update(dtype='uint32', compress='deflate', predictor=2)
with rasterio.open("labels.tif", "w", **prof) as dst:
    dst.write(labels, 1)

os.makedirs("crowns", exist_ok=True)
mask_prof = prof.copy(); mask_prof.update(dtype='uint8')
for lbl in range(1, labels.max() + 1):
    m = (labels == lbl)
    if m.sum() == 0:  # empty crown, skip
        continue
    with rasterio.open(f"crowns/crown_{lbl:05d}.tif", "w", **mask_prof) as dst:
        dst.write(m.astype(np.uint8), 1)

# --------------------------- 10. QA print-out -----------------------------
for k, lbl in enumerate(range(1, labels.max() + 1), start=1):
    px = (labels == lbl).sum()
    if k < 10 or args.verbose:
        print(f"seed {k:3d}: crown px = {px}")
print("✓  wrote labels.tif and {0} crown masks → ./crowns/".format(labels.max()))