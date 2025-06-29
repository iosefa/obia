#!/usr/bin/env python3
"""
Binary CRF crown segmentation
  • band 1: seed logit
  • band 2: ground mask  (1 = ground / void, 0 = canopy)

Output labels
  -1 nodata / ground
   0 background (canopy but no crown)
   1 crown      (seed wins)

gco‑wrapper ≥ 3 required.
"""
import numpy as np, rasterio as rio, rasterio.warp, argparse, importlib, sys
if not hasattr(np, "float128"): np.float128 = np.longdouble
gco = importlib.import_module("gco"); sys.modules.setdefault("pygco", gco)
from gco import cut_general_graph

SAFE_EDGE = 30000           # keep edge weights < int16 max


# ───────── helpers ──────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def resample_cost(cost_path, dst_profile, H, W):
    with rio.open(cost_path) as src:
        arr = src.read(1, masked=True).astype(np.float32)
        src_arr = arr.filled(0)[np.newaxis, ...]
        dst_arr = np.empty((1, H, W), np.float32)
        rasterio.warp.reproject(
            src_arr, dst_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
            resampling=rasterio.warp.Resampling.bilinear)
    cost = dst_arr[0]
    nodata = np.isnan(cost)
    cost[nodata] = 0.0
    return cost, nodata


def build_edges(cost, lam, scale, nodata):
    h, w = cost.shape
    wh = lam * cost[:, :-1] * scale
    wv = lam * cost[:-1, :] * scale
    mH = nodata[:, :-1] | nodata[:, 1:]
    mV = nodata[:-1, :] | nodata[1:, :]
    wh[mH] = wv[mV] = 0.0
    wh = np.clip(wh, 0, SAFE_EDGE).astype(np.int32).ravel()
    wv = np.clip(wv, 0, SAFE_EDGE).astype(np.int32).ravel()
    idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
    edges = np.column_stack([np.concatenate([idx[:, :-1].ravel(),
                                             idx[:-1, :].ravel()]),
                             np.concatenate([idx[:, 1: ].ravel(),
                                             idx[1:,  :].ravel()])])
    return edges.astype(np.int32), np.concatenate([wh, wv])


# ───────── main ────────────────────────────────────────────────────
def main(a):
    # ---- read two bands only --------------------------------------
    with rio.open(a.logits) as ds:
        seed_logit = ds.read(1).astype(np.float32)
        ground     = ds.read(2).astype(np.float32)      # 0 or 1
        prof       = ds.profile
    H, W = seed_logit.shape

    # ---- cost raster → logits grid --------------------------------
    cost, nodata = resample_cost(a.cost, prof, H, W)

    # ---- probabilities --------------------------------------------
    p_seed = sigmoid(seed_logit) * (1 - ground)         # zero on ground
    p_bg   = ground + (1 - ground) * (1 - p_seed)       # complements seed
    probs  = np.stack([p_bg, p_seed], 0)                # order: bg, seed

    # ---- unary  −log P × scale ------------------------------------
    unary_f = -np.log(np.clip(probs, 1e-6, 1.0))
    unary_i = (unary_f * a.scale).astype(np.int32)
    if a.label_cost:
        unary_i[1] += a.label_cost
    unary = unary_i.transpose(1, 2, 0).reshape(-1, 2)

    # ---- edge scale ------------------------------------------------
    qU = np.percentile(unary_f[:, ~nodata], 95)
    qC = np.percentile(cost[~nodata], 95)
    edge_scale = min(SAFE_EDGE / a.lam, (qU) / (qC + 1e-9))
    if a.verbose:
        print(f"Q95‑unary {qU:.1f}  Q95‑cost {qC:.3f}  edge‑scale {edge_scale:.1f}")

    edges, weights = build_edges(cost, a.lam, edge_scale, nodata)

    # ---- graph‑cut -------------------------------------------------
    pair = np.array([[0, 1],
                     [1, 0]], np.int32)                # Potts for 2 labels
    labels = cut_general_graph(edges, weights, unary, pair,
                               n_iter=a.iter, algorithm="expansion") \
             .reshape(H, W).astype(np.int32)

    labels[nodata | (ground > 0.5)] = -1               # void / ground

    # ---- write -----------------------------------------------------
    out_meta = prof.copy()
    out_meta.update(count=1, dtype="int32", nodata=-1, compress="deflate")
    with rio.open(a.out, "w", **out_meta) as dst:
        dst.write(labels, 1)
    print(f"✓ wrote {a.out} – {np.count_nonzero(labels>0):,} crown pixels")


# ---- CLI -----------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="Binary CRF crowns (seed logit + ground mask)"
    )
    pa.add_argument("logits") ; pa.add_argument("cost") ; pa.add_argument("out")
    pa.add_argument("--scale", type=float, default=300, help="× for unary")
    pa.add_argument("--lam",   type=float, default=12,  help="edge λ")
    pa.add_argument("--label-cost", type=int, default=0,
                   help="penalty for choosing the seed label")
    pa.add_argument("--iter",  type=int, default=5, help="α‑expansion iters")
    pa.add_argument("--verbose", action="store_true")
    main(pa.parse_args())