#!/usr/bin/env python3
"""
wv_sharpness_map.py  – Create a Variance-of-Laplacian sharpness map
                       from a multiband WorldView GeoTIFF.

Usage
-----
python wv_sharpness_map.py \
       --in  worldview_multiband.tif \
       --out laplacian_sharpness.tif \
       --win 31
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter            # SciPy ≥1.8
import cv2                                          # OpenCV ≥4.x


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def rgb_to_gray(rgb):
    """ITU-R 601 grayscale (expects float32 array in [0,1])."""
    coeffs = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb * coeffs).sum(axis=-1)


def variance_of_laplacian(gray, win):
    """Local variance of the 3×3 Laplacian, window = win×win pixels."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mean   = uniform_filter(lap,        size=win)
    mean2  = uniform_filter(lap * lap,  size=win)
    return mean2 - mean ** 2           # σ² = E[X²] − E[X]²


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(in_path, out_path, win):
    with rasterio.open(in_path) as src:
        meta = src.meta.copy()

        # 1. Read visible bands (Blue, Green, Red for WorldView)
        vis_bands = [2, 3, 5]
        arr = src.read(vis_bands).astype(np.float32)

        # 2. Normalise each band to [0,1]  ——  *** FIX HERE ***
        band_min = arr.min(axis=(1, 2), keepdims=True)
        band_rng = np.ptp(arr, axis=(1, 2), keepdims=True) + 1e-8
        arr = (arr - band_min) / band_rng

        # 3. Grayscale → Laplacian-variance sharpness
        arr = np.transpose(arr, (1, 2, 0))          # (rows, cols, bands)
        gray  = rgb_to_gray(arr)
        sharp = variance_of_laplacian(gray, win)

        # 4. Stretch to 0–1 for display
        lo, hi = np.percentile(sharp, [2, 98])
        sharp  = np.clip((sharp - lo) / (hi - lo), 0, 1)

        # 5. Write out
        meta.update(dtype="float32", count=1, nodata=None)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(sharp.astype("float32"), 1)

    print(f"✔ Sharpness map written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True, help="Input multiband GeoTIFF")
    ap.add_argument("--out", dest="outp", required=True, help="Output sharpness GeoTIFF")
    ap.add_argument("--win", type=int, default=31,   help="Sliding-window size (odd int)")
    args = ap.parse_args()

    if args.win < 3 or args.win % 2 == 0:
        ap.error("--win must be an odd integer ≥ 3")

    main(Path(args.inp).expanduser(), Path(args.outp).expanduser(), args.win)