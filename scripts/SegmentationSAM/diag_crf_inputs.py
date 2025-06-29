#!/usr/bin/env python3
"""
diag_crf_inputs.py
──────────────────
Inspect one seed-logit raster, its derived probability, the ground mask,
and a boundary-cost raster so we can see why the CRF behaves oddly.

Usage
-----
python diag_crf_inputs.py  <tile_dir>  <cost_raster.tif>

Example
-------
python diag_crf_inputs.py \
    /path/to/sam_logits/r0c0 \
    /path/to/cost_tiles/cost_r0_c0.tif
"""

import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio

CLIP = 12       # logit clip for numerical safety
BINS = 256      # histogram bins


def main(tile_dir: Path, cost_path: Path):
    # pick the first seed_* file just for a quick look
    seed_file = sorted(tile_dir.glob("seed_*.tif"))[0]

    with rio.open(seed_file) as ds:
        logit  = ds.read(1).astype("float32")
        ground = ds.read(2).astype("float32")

    prob = 1 / (1 + np.exp(-np.clip(logit, -CLIP, CLIP)))

    # summary JSON
    summary = {
        "seed raster": seed_file.name,
        "logit": {
            "min": float(logit.min()),
            "max": float(logit.max()),
            "mean": float(logit.mean())
        },
        "probability": {
            "min": float(prob.min()),
            "max": float(prob.max()),
            "mean": float(prob.mean())
        },
        "ground canopy fraction": float((ground < 0.5).mean())
    }
    print(json.dumps(summary, indent=2))

    # cost raster
    with rio.open(cost_path) as ds:
        cost = ds.read(1, masked=True).astype("float32").filled(0)
    print("cost  min/max/mean:",
          float(cost.min()), float(cost.max()), float(cost.mean()))

    # quick histograms
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.hist(logit.ravel(), bins=BINS, color="steelblue")
    plt.title("logit (band-1)")

    plt.subplot(1, 3, 2)
    plt.hist(prob.ravel(), bins=BINS, color="seagreen")
    plt.title("probability")

    plt.subplot(1, 3, 3)
    plt.hist(cost.ravel(), bins=BINS, color="slategray")
    plt.title("boundary cost")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python diag_crf_inputs.py <tile_dir> <cost_raster.tif>")

    td  = Path(sys.argv[1])
    cr  = Path(sys.argv[2])
    if not td.is_dir():
        sys.exit("tile_dir must be a directory containing seed_*.tif files")
    if not cr.is_file():
        sys.exit("cost raster not found")

    main(td, cr)

# #!/usr/bin/env python3
# """
# diag_crf_inputs.py – print statistics used by the CRF
# ──────────────────────────────────────────────────────
# Run *once* on a tile directory to understand why crowns disappear.
# """
#
# from pathlib import Path
# import numpy as np, rasterio as rio
# import json, textwrap, sys
#
# TILE = Path(sys.argv[1])              # e.g. sam_logits/r0c0
# COST = Path(sys.argv[2])              # boundary cost raster
# SAMPLE = 50_000                       # pixels to sample for histograms
# CHM_GROUND = 3.0
# EPS = 1e-6
#
# # ─── read shared layers ────────────────────────────────────────────
# seed0 = sorted(TILE.glob("seed_*.tif"))[0]
# with rio.open(seed0) as ds:
#     ground = ds.read(2).astype(np.float32)
#     chm    = ds.read(3).astype(np.float32)
#     H, W   = ground.shape
#
# # ─── sample pixels uniformly ───────────────────────────────────────
# rng = np.random.default_rng(0)
# ys = rng.integers(0, H, SAMPLE)
# xs = rng.integers(0, W, SAMPLE)
#
# # background probability
# p_bg = np.clip((CHM_GROUND - chm)/CHM_GROUND, 0, 1)
# p_bg = np.minimum(p_bg, 0.4)
#
# # ─── gather per‑seed stats (first 20 seeds is enough) ──────────────
# stats = []
# for f in sorted(TILE.glob("seed_*.tif"))[:20]:
#     with rio.open(f) as ds:
#         logit = ds.read(1).astype(np.float32)
#     p      = 1 / (1 + np.exp(-np.clip(logit, -12, 12)))
#     p_max  = p.max()
#     p_med  = np.median(p)
#     stats.append((f.name, p_max, p_med))
#
# # ─── boundary cost distribution ────────────────────────────────────
# with rio.open(COST) as ds:
#     cost = ds.read(1, masked=True).astype(np.float32).filled(0)
# c_sample = cost[ys, xs]
#
# # ─── print summary ─────────────────────────────────────────────────
# out = {
#     "seed_max & median (first 20 files)": [
#         {"file": n, "max": float(mx), "median": float(md)}
#         for n, mx, md in stats
#     ],
#     "background p_bg": {
#         "min": float(p_bg.min()), "max": float(p_bg.max()),
#         "median": float(np.median(p_bg))
#     },
#     "boundary cost": {
#         "min": float(c_sample.min()), "max": float(c_sample.max()),
#         "median": float(np.median(c_sample))
#     }
# }
# print(json.dumps(out, indent=2))