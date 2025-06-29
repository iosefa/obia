#!/usr/bin/env python3
"""
optimise StarDist prob / NMS thresholds from the validation set
writes:  models/crown_stardist/thresholds.json
"""

import glob, math, tifffile, numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist.utils import optimize_thresholds

# ─── utility copied from training script ────────────────────────────────────
def to_four_channels(arr):
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 4:
        return arr
    if arr.shape[-1] == 3:
        pad = np.zeros_like(arr[..., :1])
        return np.concatenate([arr, pad], axis=-1)
    if arr.shape[-1] == 1:
        return np.repeat(arr, 4, axis=-1)
    raise ValueError

# ─── load validation tiles (same logic you used during training) ────────────
VAL_IMGS = sorted(glob.glob("data/val/images/*_comp.tif"))

X_val, Y_val = [], []
for ipath in VAL_IMGS:
    mpath = ipath.replace("images", "masks").replace("_comp.tif", "_mask.tif")
    img   = tifffile.imread(ipath).astype("float32")
    img   = normalize(to_four_channels(img), 0, 1)
    mask  = tifffile.imread(mpath).astype("uint8")
    X_val.append(img)
    Y_val.append(mask)

print(f"optimising thresholds on {len(X_val)} validation tiles…")

# ─── run optimiser ─────────────────────────────────────────────────────────
model = StarDist2D(None, name="crown_stardist", basedir="models")
optimize_thresholds(model, X_val, Y_val)   # writes thresholds.json

print("✓  thresholds.json written – ready for inference")