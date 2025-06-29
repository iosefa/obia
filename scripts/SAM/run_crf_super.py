#!/usr/bin/env python
"""
Seed-guided CRF crown segmentation – superpixel CRF
--------------------------------------------------
* Gaussian blur (σ = --blur) smooths noisy seed maps
* SLIC superpixels (--superpixels) ≈ 2 000 per 512×512 tile
  → faster and sharper crowns than pixel-grid CRF
Outputs
  labels.tif                 uint32 crown IDs   (0 = background)
  crowns/crown_#####.tif     one uint8 mask per crown
"""
import glob, os, sys, argparse, importlib, warnings
import numpy as np, rasterio
from rasterio.enums import Resampling
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from skimage.segmentation import slic
from skimage.util import img_as_float

# ---------- pygco ----------------------------------------------------------
if not hasattr(np, "float128"):
    np.float128 = np.longdouble
gco = importlib.import_module("gco"); sys.modules.setdefault("pygco", gco)
from gco import cut_general_graph

# ---------- CLI ------------------------------------------------------------
cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cli.add_argument("logit_dir")
cli.add_argument("cost_tif")
cli.add_argument("chm_tif")                             # dummy path OK
cli.add_argument("--scale",  type=float, default=50,    help="unary scale")
cli.add_argument("--lam",    type=float, default=12,    help="λ smoothness")
cli.add_argument("--beta",   type=float, default=0.10,  help="background cap")
cli.add_argument("--blur",   type=float, default=2,     help="Gaussian σ")
cli.add_argument("--superpixels", type=int, default=2000,
                 help="≈ number of SLIC segments (0 → pixel grid)")
cli.add_argument("--iter",   type=int,   default=5,     help="α/β-swap iters")
cli.add_argument("--seed_tol", type=float, default=0.02,
                 help="drop seeds whose max(prob) ≤ tol")
cli.add_argument("--verbose", action="store_true")
args = cli.parse_args()

# ---------- helpers --------------------------------------------------------
def read_band(path, like=None, res=Resampling.bilinear):
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(1, like.height, like.width),
            resampling=res) if like else src.read(1)
    return data.astype(np.float32)

def add_edge(buf, a, b, w):
    if a == b:
        return
    if a > b:                                  # canonical order
        a, b = b, a
    s = buf.get((a, b))
    if s is None:
        buf[(a, b)] = [w, 1]
    else:
        s[0] += w
        s[1] += 1

# ---------- 1. cost & CHM gradient ----------------------------------------
with rasterio.open(args.cost_tif) as cost_src:
    cost = cost_src.read(1).astype(np.float32)
    H, W = cost_src.height, cost_src.width
    prof = cost_src.profile.copy()

try:                                            # optional CHM gradient
    chm  = read_band(args.chm_tif, like=cost_src)
    grad = gaussian_gradient_magnitude(chm, sigma=1)
    cost = 0.5 * (cost + grad)
except Exception as e:
    warnings.warn(f"CHM gradient skipped: {e}")

# ---------- 2. load seed probability rasters ------------------------------
paths = sorted(glob.glob(os.path.join(args.logit_dir, "*.tif")))
if not paths:
    sys.exit(f"No *.tif files in {args.logit_dir}")

probs = np.empty((len(paths), H, W), np.float32)
for i, pth in enumerate(tqdm(paths, desc="loading seeds")):
    probs[i] = read_band(pth, like=cost_src)

if probs.min() < 0 or probs.max() > 1:           # logits → probs
    probs = 1.0 / (1.0 + np.exp(-probs))

probs = np.stack([gaussian_filter(p, sigma=args.blur, truncate=3.0)
                  for p in probs])               # smooth noise

keep  = np.where(probs.max(axis=(1, 2)) > args.seed_tol)[0]
probs = probs[keep]
n_seeds = len(keep)
if n_seeds == 0:
    sys.exit("All seeds empty after thresholding")

# ---------- 3. superpixels (or pixel grid) --------------------------------
if args.superpixels <= 0:
    seg = np.arange(H * W, dtype=np.int32).reshape(H, W)  # one px per seg
else:
    img = img_as_float((cost - cost.min()) / (np.ptp(cost) + 1e-6))
    seg = slic(img, n_segments=args.superpixels, compactness=10,
               start_label=0, convert2lab=False,
               enforce_connectivity=True, channel_axis=None)
    if args.verbose:
        print(f"SLIC produced {seg.max() + 1} superpixels")

n_nodes = seg.max() + 1
L       = n_seeds + 1                              # background + seeds

# ---------- 4. probability cube & normalise -------------------------------
bg   = np.clip(1.0 - probs.max(axis=0), 1e-6, args.beta)
cube = np.concatenate([bg[None, ...], probs], axis=0)  # (L, H, W)
cube /= cube.sum(axis=0, keepdims=True)

# ---------- 5. UNARY per superpixel ---------------------------------------
areas  = np.bincount(seg.ravel(), minlength=n_nodes).astype(np.float32)
unary  = np.zeros((n_nodes, L), np.float32)

for lbl in range(L):
    unary[:, lbl] = np.bincount(
        seg.ravel(), weights=cube[lbl].ravel(), minlength=n_nodes)

unary = unary / (areas[:, None] + 1e-6)
unary = (-np.log(unary) * args.scale).astype(np.int32)

# ----- hard-clamp seed’s own superpixel
CLAMP = 2_000_000                                 # ≥ any edge cost
for seed_idx, p in enumerate(cube[1:], start=1):
    y, x = np.unravel_index(np.argmax(p), p.shape)
    sp   = seg[y, x]
    unary[sp, :] = CLAMP
    unary[sp, seed_idx] = 0

unary = np.ascontiguousarray(unary)               # pygco safety

# ---------- 6. pair-wise Potts  -------------------------------------------
pairwise = np.ones((L, L), np.int32)
np.fill_diagonal(pairwise, 0)

# ---------- 7. build edge list --------------------------------------------
edge_buf = {}
# horizontal neighbours
ys, xs = np.nonzero(seg[:, :-1] != seg[:, 1:])
for y, x in zip(ys, xs):
    add_edge(edge_buf, seg[y, x], seg[y, x+1],
             args.lam * ((cost[y, x] + cost[y, x+1]) * 0.5) * args.scale)
# vertical neighbours
ys, xs = np.nonzero(seg[:-1, :] != seg[1:, :])
for y, x in zip(ys, xs):
    add_edge(edge_buf, seg[y, x], seg[y+1, x],
             args.lam * ((cost[y, x] + cost[y+1, x]) * 0.5) * args.scale)

edges, eweights = zip(*[((a, b), max(1, w_sum / cnt))
                        for (a, b), (w_sum, cnt) in edge_buf.items()])
edges    = np.asarray(edges,    dtype=np.int32)
eweights = np.asarray(eweights, dtype=np.int32)

if args.verbose:
    print(f"Graph nodes={n_nodes}, edges={edges.shape[0]}, labels={L}")

# ---------- 8. graph cut ---------------------------------------------------
labels_sp = cut_general_graph(edges, eweights, unary, pairwise,
                              n_iter=args.iter, algorithm='swap'
                              ).astype(np.int32)

labels = labels_sp[seg].astype(np.uint32)

# ---------- 9. save results -----------------------------------------------
prof.update(dtype='uint32', compress='deflate', predictor=2)
with rasterio.open("labels.tif", "w", **prof) as dst:
    dst.write(labels, 1)

os.makedirs("crowns", exist_ok=True)
mask_prof = prof.copy(); mask_prof.update(dtype='uint8')

for cid in range(1, labels.max() + 1):
    m = (labels == cid)
    if m.sum() == 0:
        continue
    with rasterio.open(f"crowns/crown_{cid:05d}.tif", "w", **mask_prof) as dst:
        dst.write(m.astype(np.uint8), 1)

for k, cid in enumerate(range(1, labels.max() + 1), start=1):
    if k <= 10 or args.verbose:
        print(f"crown {cid:3d}  px = {int((labels==cid).sum())}")
print(f"✓  wrote labels.tif and {labels.max()} crowns to ./crowns/")