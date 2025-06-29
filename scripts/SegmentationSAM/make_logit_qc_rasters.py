#!/usr/bin/env python3
"""
make_logit_qc_rasters.py
Create max-probability and arg-max QA rasters from a SAM-logits tile.
"""

import numpy as np, rasterio, sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
in_tile = Path(sys.argv[1])          # e.g. sam_logits_r0_c0.tif
out_dir = in_tile.parent             # same folder; change if you wish

with rasterio.open(in_tile) as src:
    logits = src.read()              # (S, H, W)
    prof   = src.profile

# 1. logits → probabilities
probs = 1.0 / (1.0 + np.exp(-logits))        # float32 (S,H,W)

# 2a. max probability composite
max_prob = probs.max(axis=0).astype("float32")          # (H,W)

# 2b. winner-takes-all label map  (background = 0)
argmax = (probs.argmax(axis=0) + 1).astype("uint16")    # (H,W)

# write rasters ------------------------------------------------
prof_max = prof.copy(); prof_max.update(count=1, dtype="float32", nodata=None)
with rasterio.open(out_dir / f"{in_tile.stem}_max_prob.tif", "w", **prof_max) as dst:
    dst.write(max_prob, 1)

prof_lbl = prof.copy(); prof_lbl.update(count=1, dtype="uint16", nodata=0,
                                        compress="lzw")
with rasterio.open(out_dir / f"{in_tile.stem}_argmax_id.tif", "w", **prof_lbl) as dst:
    dst.write(argmax, 1)

print("✓ wrote",
      f"{in_tile.stem}_max_prob.tif and {in_tile.stem}_argmax_id.tif → {out_dir}")
