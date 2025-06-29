#!/usr/bin/env python3
"""
01_make_crops.py
────────────────
Write a 1024×1024 RGB PNG + binary‑mask PNG for every crown whose
padded square window lies fully inside the raster.

RGB  = bands 7‑5‑3 stretched to uint8
Mask = 1‑channel {0,255}
"""

from pathlib import Path
import sys, warnings
import numpy as np, geopandas as gpd, rasterio
from rasterio import windows, features
from shapely.geometry import box
from PIL import Image

# ─── parameters you may tweak ───────────────────────────────────────
RGB_BANDS  = (7, 5, 3)         # WV‑3 pseudo‑RGB
RGB_PCTL   = (2, 98)           # contrast stretch
CANVAS     = 1024              # final side length (px)
PAD_NATIVE = 32                # native‑pixel padding around bbox
# ────────────────────────────────────────────────────────────────────

def u16_to_u8(arr):
    lo, hi = np.percentile(arr, RGB_PCTL)
    return np.clip((arr - lo) * 255 / (hi - lo + 1e-6), 0, 255).astype("uint8")

def main(raster, crowns, out_dir):
    raster  = Path(raster).expanduser()
    crowns  = Path(crowns).expanduser()
    out_dir = Path(out_dir).expanduser()
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(crowns)
    with rasterio.open(raster) as src:
        gdf = gdf.to_crs(src.crs)
        gdf = gdf[gdf.intersects(box(*src.bounds))]
        if gdf.empty:
            sys.exit("No polygons overlap raster.")

        written = skipped = 0
        H, W = src.height, src.width

        for geom in gdf.geometry:
            if geom.is_empty:
                continue

            # tight bbox around polygon
            minx, miny, maxx, maxy = geom.bounds
            r0, c0 = src.index(minx, maxy)
            r1, c1 = src.index(maxx, miny)
            win = windows.Window.from_slices(
                (max(r0, 0), r1 + 1),
                (max(c0, 0), c1 + 1))

            # square + native‑px padding
            side = max(win.height, win.width) + 2 * PAD_NATIVE
            row_off = int(win.row_off + (win.height - side) // 2)
            col_off = int(win.col_off + (win.width  - side) // 2)

            # ── skip if the square would poke outside the raster ──
            if row_off < 0 or col_off < 0 or \
               row_off + side > H or col_off + side > W:
                skipped += 1
                continue

            win_sq = windows.Window(col_off, row_off, side, side)

            rgb16 = src.read(RGB_BANDS, window=win_sq).astype("float32")
            rgb8  = np.stack([u16_to_u8(b) for b in rgb16], 0)

            mask = features.geometry_mask(
                [geom], out_shape=(side, side),
                transform=src.window_transform(win_sq),
                invert=True, all_touched=False).astype("uint8") * 255

            canvas = np.zeros((CANVAS, CANVAS, 3), np.uint8)
            m_can  = np.zeros((CANVAS, CANVAS),    np.uint8)
            off_r  = (CANVAS - side) // 2
            off_c  = (CANVAS - side) // 2
            canvas[off_r:off_r + side, off_c:off_c + side] = np.moveaxis(rgb8, 0, -1)
            m_can [off_r:off_r + side, off_c:off_c + side] = mask

            stem = f"crown_{written:05d}"
            Image.fromarray(canvas).save(out_dir / "images" / f"{stem}.png")
            Image.fromarray(m_can ).save(out_dir / "masks"  / f"{stem}_mask.png")
            written += 1

    print(f"✓ kept {written} crowns   •   skipped {skipped} touching the edge")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster",   required=True)
    ap.add_argument("--crowns",   required=True)
    ap.add_argument("--out-dir",  required=True)
    args = ap.parse_args()
    main(args.raster, args.crowns, args.out_dir)