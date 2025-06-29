#!/usr/bin/env python3
"""
make_cost_surface.py  (REV-2 · vector-SLIC aware)
────────────────────────────────────────────────────────────
Build a single-band boundary-cost raster (0–1; nodata = –9999)
from
  1. CHM gradient magnitude
  2. 1 – NDVI  (canopy gaps)
  3. Panchromatic texture entropy
  4. SLIC edge strength  (raster **or** GPKG polygons)

Weights default to 0.35 / 0.25 / 0.25 / 0.15.

Example
-------
python make_cost_surface.py \
    --wv3  tile_8band.tif \
    --chm  tile_chm.tif   \
    --slic tile_segments.gpkg \
    --out  tile_cost.tif
"""
from pathlib import Path
import sys, warnings

import click
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy.ndimage import sobel
from skimage.filters.rank import entropy
from skimage.morphology import disk

# optional – only needed if --slic is a GeoPackage
try:
    import geopandas as gpd
except ImportError:
    gpd = None

# ───────────── low-level helpers ────────────────────────────
def read_band(path, idx=1):
    with rasterio.open(path) as src:
        arr = src.read(idx, masked=True).astype(np.float32)
        prof = src.profile.copy()
    return arr.filled(np.nan), prof


def normalise(arr):
    lo, hi = np.nanpercentile(arr, (2, 98))
    arr_clip = np.clip(arr, lo, hi)
    with np.errstate(invalid="ignore"):
        out = (arr_clip - lo) / (hi - lo)
    return np.nan_to_num(out)


def chm_gradient(chm):
    dx = sobel(chm, axis=1, mode="nearest")
    dy = sobel(chm, axis=0, mode="nearest")
    return normalise(np.hypot(dx, dy))


def ndvi(red, nir):
    return np.clip((nir - red) / (nir + red + 1e-9), -1, 1)


def texture_entropy(pan):
    pan_u8 = (normalise(pan) * 255).astype(np.uint8)
    return normalise(entropy(pan_u8, disk(3)))      # 7 × 7 window


def slic_edge(label_img):
    edge = np.zeros_like(label_img, dtype=np.uint8)
    edge[:-1, :] |= label_img[:-1, :] != label_img[1:, :]
    edge[:, :-1] |= label_img[:, :-1] != label_img[:, 1:]
    return normalise(edge.astype(np.float32))


def rasterise_slic_gpkg(gpkg_path, tgt_profile):
    """Rasterise polygons (segment_id) onto WV-3 grid."""
    if gpd is None:
        raise SystemExit("geopandas is required for vector SLIC input.")

    west, south, east, north = tgt_profile["bounds"]
    gdf = gpd.read_file(gpkg_path, bbox=(west, south, east, north))

    if gdf.empty:
        raise SystemExit("SLIC GPKG has no polygons over this tile.")

    if gdf.crs != tgt_profile["crs"]:
        gdf = gdf.to_crs(tgt_profile["crs"])

    shapes = []
    for geom, seg in zip(gdf.geometry, gdf["segment_id"]):
        try:
            seg_id = int(seg)
        except Exception:
            continue
        if geom is not None and not geom.is_empty:
            shapes.append((geom, seg_id))

    if not shapes:
        raise SystemExit("No valid SLIC polygons with 'segment_id' found.")

    H, W = tgt_profile["height"], tgt_profile["width"]
    lbl = rasterize(
        shapes,
        out_shape=(H, W),
        transform=tgt_profile["transform"],
        fill=0,
        dtype=np.uint32,
        all_touched=False,
    )
    return lbl
# ───────────────────────────── CLI ──────────────────────────
@click.command()
@click.option("--wv3",  type=click.Path(exists=True), required=True,
              help="8-band WV-3 stack (C,B,G,Y,R,RE,NIR1,NIR2).")
@click.option("--chm",  type=click.Path(exists=True), required=True,
              help="CHM GeoTIFF aligned to WV-3 tile.")
@click.option("--slic", type=click.Path(exists=True), default=None,
              help="Raster SLIC TIFF **or** GeoPackage with polygons "
                   "(must have integer 'segment_id').")
@click.option("--out",  type=click.Path(), required=True,
              help="Output cost TIFF.")
@click.option("--weights", nargs=4, type=float,
              default=(0.35, 0.25, 0.25, 0.15), show_default=True,
              help="Weights: grad gap texture slic.")
def main(wv3, chm, slic, out, weights):
    w_grad, w_gap, w_tex, w_slic = weights
    if abs(sum(weights) - 1) > 1e-6:
        raise SystemExit("Weights must sum to 1.")

    # 1 ─ read WV-3 bands ------------------------------------------
    with rasterio.open(wv3) as src:
        C, B, G, Y, R, RE, N1, N2 = src.read(masked=True).astype(np.float32)
        profile = src.profile.copy()
        profile["bounds"] = rasterio.transform.array_bounds(
            profile["height"], profile["width"], profile["transform"]
        )

    # 2 ─ CHM -------------------------------------------------------
    chm_arr, _ = read_band(chm)

    # 3 ─ compute layers -------------------------------------------
    grad = chm_gradient(chm_arr)
    gap  = normalise(1 - ndvi(R, N1))
    tex  = texture_entropy(C)

    if slic:
        if slic.lower().endswith(".gpkg"):
            slic_lab = rasterise_slic_gpkg(slic, profile)
        else:  # assume raster label image
            slic_lab, _ = read_band(slic)
        edge = slic_edge(slic_lab)
    else:
        edge = 0.0
        # re-scale weights if SLIC missing
        s = w_grad + w_gap + w_tex
        w_grad, w_gap, w_tex, w_slic = (
            w_grad / s, w_gap / s, w_tex / s, 0.0
        )
        warnings.warn("No SLIC provided – cost built from 3 terms only.")

    cost = (
        w_grad * grad +
        w_gap  * gap  +
        w_tex  * tex  +
        w_slic * edge
    )
    cost = np.clip(cost, 0, 1).astype(np.float32)

    # 4 ─ write -----------------------------------------------------
    nodata_val = -9999.0
    cost[np.isnan(cost)] = nodata_val
    profile.update(count=1, dtype="float32",
                   compress="deflate", nodata=nodata_val)
    out_path = Path(out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cost, 1)

    click.echo(f"✓ cost surface written → {out_path} (nodata={nodata_val})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

# #!/usr/bin/env python3
# """
# make_cost_surface.py  (REV-1.1)
# ────────────────────────────────────────────────────────────
# Build a single-band **boundary-cost raster** (0–1; nodata = −9999)
# from:
#   1. CHM gradient magnitude (edges at height drops)
#   2. 1 – NDVI             (penalise canopy gaps)
#   3. Panchromatic texture entropy
#   4. SLIC super-pixel edge strength  (optional)
#
# Weights default to 0.35 / 0.25 / 0.25 / 0.15 but can be changed.
#
# Example
# -------
# python make_cost_surface.py \
#     --wv3   tile_8band.tif \
#     --chm   tile_chm.tif   \
#     --slic  tile_slic.tif  \
#     --out   tile_cost.tif  \
#     --weights 0.35 0.25 0.25 0.15
# """
# from pathlib import Path
# import sys
#
# import click
# import numpy as np
# import rasterio
# from rasterio.enums import Resampling
# from scipy.ndimage import sobel
# from skimage.filters.rank import entropy
# from skimage.morphology import disk
#
# # ────────────────────────── utils ─────────────────────────────
#
# def read_band(path, idx=1):
#     with rasterio.open(path) as src:
#         arr = src.read(idx, masked=True).astype(np.float32)
#         profile = src.profile.copy()
#     return arr.filled(np.nan), profile
#
#
# def normalise(arr):
#     m, M = np.nanpercentile(arr, (2, 98))
#     arr_clip = np.clip(arr, m, M)
#     with np.errstate(invalid="ignore"):
#         out = (arr_clip - m) / (M - m)
#     return np.nan_to_num(out)
#
#
# def chm_gradient(chm):
#     dx = sobel(chm, axis=1, mode="nearest")
#     dy = sobel(chm, axis=0, mode="nearest")
#     return normalise(np.hypot(dx, dy))
#
#
# def ndvi(red, nir):
#     nd = (nir - red) / (nir + red + 1e-9)
#     return np.clip(nd, -1, 1)
#
#
# def texture_entropy(pan):
#     pan_u8 = (normalise(pan) * 255).astype(np.uint8)
#     ent = entropy(pan_u8, disk(3))  # 7×7 window
#     return normalise(ent)
#
#
# def slic_edge(slic):
#     edge = np.zeros_like(slic, dtype=np.uint8)
#     edge[:-1, :] |= slic[:-1, :] != slic[1:, :]
#     edge[:, :-1] |= slic[:, :-1] != slic[:, 1:]
#     return normalise(edge.astype(np.float32))
#
# # ────────────────────────── CLI ──────────────────────────────
#
# @click.command()
# @click.option("--wv3",  type=click.Path(exists=True), required=True,
#               help="8-band WV-3 stack (C,B,G,Y,R,RE,NIR1,NIR2).")
# @click.option("--chm",  type=click.Path(exists=True), required=True,
#               help="CHM GeoTIFF aligned to WV-3 tile.")
# @click.option("--slic", type=click.Path(exists=True), default=None,
#               help="Optional SLIC label raster (uint32).")
# @click.option("--out",  type=click.Path(), required=True,
#               help="Output cost TIFF.")
# @click.option("--weights", nargs=4, type=float,
#               default=(0.35, 0.25, 0.25, 0.15), show_default=True,
#               help="Weights: grad gap texture slic.")
#
# def main(wv3, chm, slic, out, weights):
#     w_grad, w_gap, w_tex, w_slic = weights
#     if abs(sum(weights) - 1) > 1e-6:
#         raise SystemExit("Weights must sum to 1.")
#
#     # 1 ─ read WV-3 bands -----------------------------------------
#     with rasterio.open(wv3) as src:
#         C, B, G, Y, R, RE, N1, N2 = src.read(masked=True).astype(np.float32)
#         profile = src.profile.copy()
#
#     # 2 ─ CHM ------------------------------------------------------
#     chm_arr, _ = read_band(chm)
#
#     # 3 ─ compute layers ------------------------------------------
#     grad = chm_gradient(chm_arr)
#     gap  = normalise(1 - ndvi(R, N1))  # use NIR1
#     tex  = texture_entropy(C)
#
#     if slic:
#         slic_arr, _ = read_band(slic)
#         edge = slic_edge(slic_arr)
#     else:
#         edge = 0.0
#         # renormalise weights if SLIC missing
#         s = w_grad + w_gap + w_tex
#         w_grad, w_gap, w_tex = [w / s for w in (w_grad, w_gap, w_tex)]
#         w_slic = 0.0
#
#     cost = (w_grad * grad + w_gap * gap + w_tex * tex + w_slic * edge)
#     cost = np.clip(cost, 0, 1).astype(np.float32)
#
#     # 4 ─ write with nodata ---------------------------------------
#     nodata_val = -9999.0
#     cost_np = np.where(np.isnan(cost), nodata_val, cost)
#
#     profile.update(count=1, dtype="float32", compress="deflate")
#     out_path = Path(out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with rasterio.open(out_path, "w", **profile) as dst:
#         dst.write(cost_np, 1)
#         dst.nodata = nodata_val  # ensure GDAL recognises nodata
#
#     click.echo(f"✓ cost surface written → {out_path} (nodata={nodata_val})")
#
#
# if __name__ == "__main__":
#     try:
#         main()  # pylint: disable=no-value-for-parameter
#     except Exception as exc:
#         click.echo(f"Error: {exc}", err=True)
#         sys.exit(1)
