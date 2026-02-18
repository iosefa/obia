from pathlib import Path
import warnings

import click
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import sobel
from skimage.filters.rank import entropy
from skimage.morphology import disk
import geopandas as gpd


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
    return normalise(entropy(pan_u8, disk(3)))


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


def make_cost_surface(wv3, chm, out, slic=None, weights=(0.5, 0.25, 0.25, 0)):
    w_grad, w_gap, w_tex, w_slic = weights
    if abs(sum(weights) - 1) > 1e-6:
        raise SystemExit("Weights must sum to 1.")

    with rasterio.open(wv3) as src:
        C, B, G, Y, R, RE, N1, N2 = src.read(masked=True).astype(np.float32)
        profile = src.profile.copy()
        profile["bounds"] = rasterio.transform.array_bounds(
            profile["height"], profile["width"], profile["transform"]
        )

    chm_arr, _ = read_band(chm)

    grad = chm_gradient(chm_arr)
    gap  = normalise(1 - ndvi(R, N1))
    tex  = texture_entropy(C)

    if slic:
        if slic.lower().endswith(".gpkg"):
            slic_lab = rasterise_slic_gpkg(slic, profile)
        else:
            slic_lab, _ = read_band(slic)
        edge = slic_edge(slic_lab)
    else:
        edge = 0.0
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

    nodata_val = -9999.0
    cost[np.isnan(cost)] = nodata_val
    profile.update(count=1, dtype="float32",
                   compress="deflate", nodata=nodata_val)
    out_path = Path(out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cost, 1)

    click.echo(f"✓ cost surface written → {out_path} (nodata={nodata_val})")
