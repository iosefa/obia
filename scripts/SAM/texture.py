#!/usr/bin/env python3
"""
compute_texture.py – extract an entropy‑based texture layer from a GeoTIFF.

• Reads *image.tif* (or any input you specify).
• Uses skimage’s rank‑entropy with a disk‑shaped window.
• Writes *texture.tif* with the same georeferencing.

Example
-------
    python compute_texture.py                # defaults: image.tif → texture.tif
    python compute_texture.py -i img.tif -o tex.tif -r 5 -b 3

Dependencies
------------
    pip install rasterio scikit-image numpy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from skimage.filters.rank import entropy
from skimage.morphology import disk


def _normalise(arr: np.ndarray, p_low: int = 2, p_high: int = 98) -> np.ndarray:
    """Robust linear stretch to [0,1] via percentile clipping."""
    lo, hi = np.nanpercentile(arr, (p_low, p_high))
    arr = np.clip(arr, lo, hi)
    with np.errstate(invalid="ignore"):
        out = (arr - lo) / (hi - lo)
    return np.nan_to_num(out)


def _texture_entropy(band: np.ndarray, radius: int = 3) -> np.ndarray:
    """Compute entropy texture and re‑normalise to [0,1]."""
    band_u8 = (_normalise(band) * 255).astype(np.uint8)
    tex = entropy(band_u8, disk(radius))
    return _normalise(tex)


def main():
    parser = argparse.ArgumentParser(description="Compute entropy texture from a GeoTIFF band")
    parser.add_argument("--input", "-i", default="image.tif", help="Input GeoTIFF path")
    parser.add_argument("--output", "-o", default="texture.tif", help="Output GeoTIFF path")
    parser.add_argument("--radius", "-r", type=int, default=3, help="Entropy kernel radius in pixels (disk)")
    parser.add_argument("--band", "-b", type=int, default=1, help="1‑based band index to use for texture")
    args = parser.parse_args()

    in_path: Path = Path(args.input)
    out_path: Path = Path(args.output)

    # 1. Read the requested band
    with rasterio.open(in_path) as src:
        band = src.read(args.band, masked=True).astype(np.float32).filled(np.nan)
        profile = src.profile.copy()

    # 2. Compute texture
    texture = _texture_entropy(band, radius=args.radius).astype(np.float32)

    # 3. Write GeoTIFF
    profile.update(count=1, dtype="float32", nodata=None, compress="deflate")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(texture, 1)

    print(f"Texture written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
