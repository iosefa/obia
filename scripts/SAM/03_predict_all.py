#!/usr/bin/env python3
"""
03_predict_all.py  –  probability mosaic for *every* seed in seeds_cost.gpkg

Outputs
───────
SAM/
  tmp_prob/seed_<cluster>.tif   – per‑seed probability patches   (temporary)
  crowns_prob.tif               – full‑scene, max‑mosaicked probability
"""

from __future__ import annotations
from pathlib import Path
import argparse, random, math, shutil
import numpy as np, cv2, rasterio, torch, geopandas as gpd
from rasterio.transform import rowcol, Affine
from shapely.geometry import box as Box, Point
from segment_anything import sam_model_registry
from samgeo import SamGeo

# ─── static paths ─────────────────────────────────────────────────
ROOT   = Path(__file__).parent
IMG    = ROOT / "image2.tif"
CHM    = ROOT / "chm2.tif"
SEEDS  = ROOT / "seeds_cost2.gpkg"

SAM_META = ROOT / "sam_vit_l_0b3195.pth"
SAM_DEC  = ROOT / "sam_crowns_decoder.pth"

# temp dir + final mosaic
TMP_DIR   = ROOT / "tmp_prob2"
OUT_MOSAIC= ROOT / "crowns_prob2.tif"

# ─── inference hyper‑params (same for every crown) ────────────────
CONTEXT_FCTR  = 1.5
CHM_GROUND    = 3.0
USE_CHM       = True

# BOX_METRE     = 12.0
# SIGMA_METRE   = 1.0
# FG_EXTRA      = 40               # <- higher density of positive points
# NEG_MAX_GND   = 150
# NEG_MAX_CAN   = 100

# Tweaks for dense-crown areas
BOX_METRE    = 8.0            # was 12
SIGMA_METRE  = 0.5            # sharper height map
FG_EXTRA     = 80             # was 40
NEG_MAX_CAN  = 300            # was 100
NEG_MAX_GND  = 150
CHM_BOOST    = 3.0            # weight in logit += …

# Gaussian “core”
CORE_SIGMA = BOX_METRE / 2.0     # metres

# ─── small helpers ────────────────────────────────────────────────
def stretch(arr, p2=2, p98=98):
    lo, hi = np.percentile(arr, (p2, p98))
    return np.clip((arr - lo) * 255 / (hi - lo + 1e-9),
                   0, 255).astype("uint8")

def gauss(a, σ_pix):
    k = int(round(σ_pix * 3)) * 2 + 1
    return cv2.GaussianBlur(a, (k, k), σ_pix,
                            borderType=cv2.BORDER_REPLICATE)

def resize_long(img, long=1024):
    h, w = img.shape[:2]
    s = long / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)),
                      interpolation=cv2.INTER_LINEAR), s

# ─── load once – big stuff in memory ‐─────────────────────────────
with rasterio.open(IMG) as src:
    IMG_TFM, IMG_CRS = src.transform, src.crs
    RES = abs(src.transform.a)
    RGB16 = np.stack([src.read(i) for i in (7, 5, 3)])
    IMG_H, IMG_W = src.height, src.width

with rasterio.open(CHM) as src_chm:
    CHM_FULL = src_chm.read(1)        # float32  (same grid)

# SAM model loaded once
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
SAM = sam_model_registry["vit_l"](checkpoint=str(SAM_META)).to(DEVICE)
SAM.load_state_dict(torch.load(SAM_DEC, map_location=DEVICE), strict=False)
SAM.mask_decoder.num_multimask_outputs = 1
SAM.mask_decoder.mask_tokens = torch.nn.Embedding(
    1, SAM.mask_decoder.mask_tokens.embedding_dim, device=DEVICE)
PREDICTOR = SamGeo(model=SAM, automatic=False, device=DEVICE)

# ─── per‑seed inference routine ───────────────────────────────────
def process_seed(cluster_id: int, out_path: Path):
    # locate seed centre (take the first point of the cluster)
    seeds = gpd.read_file(SEEDS).to_crs(IMG_CRS)
    geom   = seeds.loc[seeds.cluster == cluster_id, "geometry"].iloc[0]
    cx, cy = geom.x, geom.y
    row, col = rowcol(IMG_TFM, cx, cy)

    # RGB patch
    half_pad = int(round(CONTEXT_FCTR * BOX_METRE / RES))
    r0, r1 = max(row - half_pad, 0), min(row + half_pad, IMG_H)
    c0, c1 = max(col - half_pad, 0), min(col + half_pad, IMG_W)

    rgb_patch = np.moveaxis(RGB16[:, r0:r1, c0:c1], 0, -1)
    h_p, w_p = rgb_patch.shape[:2]
    rgb_u8 = np.stack([stretch(rgb_patch[..., i]) for i in range(3)], -1)
    rgb_1024, s = resize_long(rgb_u8)

    # prompt box & seed in resized pixels
    half_box_px = BOX_METRE / (2 * RES) * s
    rp, cp = (row - r0) * s, (col - c0) * s
    box_px = np.array([[cp - half_box_px, rp - half_box_px,
                        cp + half_box_px, rp + half_box_px]],
                      dtype="float32")

    # CHM patch
    chm_patch = CHM_FULL[r0:r1, c0:c1]

    # foreground points  – seed + FG_EXTRA tallest CHM in box
    fg = [[cp, rp]]
    ys, xs = np.where(chm_patch > CHM_GROUND)
    in_box = ((abs(xs - (col - c0)) <= BOX_METRE / (2 * RES)) &
              (abs(ys - (row - r0)) <= BOX_METRE / (2 * RES)))
    xs, ys = xs[in_box], ys[in_box]
    if xs.size:
        order = np.argsort(chm_patch[ys, xs])[::-1][:FG_EXTRA]
        for x, y in zip(xs[order], ys[order]):
            if x == (col - c0) and y == (row - r0):
                continue
            fg.append([x * s, y * s])

    pt_px  = np.asarray(fg, dtype="float32")
    lbl_px = np.ones(len(pt_px), dtype="int32")

    # negatives
    ys_g, xs_g = np.where(chm_patch < CHM_GROUND)
    neg_g = np.column_stack([xs_g * s, ys_g * s]).astype("float32")
    if len(neg_g) > NEG_MAX_GND:
        neg_g = neg_g[np.random.choice(len(neg_g), NEG_MAX_GND, False)]

    ys_c, xs_c = np.where(chm_patch > CHM_GROUND)
    neg_c = []
    if NEG_MAX_CAN:
        half_box_pix = BOX_METRE / (2 * RES)
        for x, y in zip(xs_c, ys_c):
            if (abs(x - (col - c0)) > half_box_pix or
                    abs(y - (row - r0)) > half_box_pix):
                neg_c.append([x * s, y * s])
        if len(neg_c) > NEG_MAX_CAN:
            neg_c = random.sample(neg_c, NEG_MAX_CAN)
    neg_np = np.asarray(neg_c, dtype="float32") if neg_c else \
             np.empty((0, 2), "float32")

    pts = np.vstack([pt_px, neg_g, neg_np])
    lbl = np.hstack([lbl_px,
                     np.zeros(len(neg_g) + len(neg_np), "int32")])

    # SAM forward
    PREDICTOR.set_image(rgb_1024)
    logit_rs = PREDICTOR.predict(
        point_coords=pts, point_labels=lbl, box=box_px,
        tile=False, return_logits=True, multimask_output=False,
        return_results=True)[2][0]
    logit = cv2.resize(logit_rs, (w_p, h_p), cv2.INTER_LINEAR)

    # CHM gentle prior
    try:
        if USE_CHM:
            σ_pix = SIGMA_METRE / RES
            chm_s = gauss(chm_patch.astype("float32"), σ_pix)
            h_seed = chm_patch[int(row - r0), int(col - c0)]
            dh = np.clip(chm_s - h_seed, 0, None)
            logit += CHM_BOOST * dh / (dh.max() + 1e-6)

    except IndexError as e:
        print(f"WARNING seed {cid}: {e} (probably out of bounds)")

    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -12, 12)))

    # Gaussian core
    yy, xx = np.indices(prob.shape)
    d_m = np.hypot(yy - (row - r0), xx - (col - c0)) * RES
    core_w = np.exp(-(d_m ** 2) / (2 * CORE_SIGMA ** 2))
    prob *= core_w / core_w.max()

    # write patch GeoTIFF (nodata = 0)
    meta = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": h_p, "width": w_p, "crs": IMG_CRS,
        "transform": Affine(IMG_TFM.a, IMG_TFM.b, IMG_TFM.c + c0 * IMG_TFM.a,
                            IMG_TFM.d, IMG_TFM.e, IMG_TFM.f + r0 * IMG_TFM.e),
        "compress": "deflate", "nodata": 0.0
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(prob.astype("float32"), 1)


# ─── script entry ‑point ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true",
                        help="delete tmp_prob/ after mosaic creation")
    args = parser.parse_args()

    TMP_DIR.mkdir(exist_ok=True, parents=True)

    clusters = gpd.read_file(SEEDS)["cluster"].unique()
    print(f"Processing {len(clusters):,} crowns …")

    for cid in clusters:
        out_seed = TMP_DIR / f"seed_{cid}.tif"
        if out_seed.exists():
            continue               # reuse existing patch if present
        process_seed(int(cid), out_seed)
        print(f"  • seed {cid}")

    # ── build full mosaic (max of all patches) ─────────────────────
    mosaic = np.zeros((IMG_H, IMG_W), dtype="float32")

    for tif in TMP_DIR.glob("seed_*.tif"):
        with rasterio.open(tif) as src:
            data = src.read(1)
            r0, c0 = rowcol(IMG_TFM, src.bounds.left, src.bounds.top)
            mosaic[r0:r0 + src.height, c0:c0 + src.width] = np.maximum(
                mosaic[r0:r0 + src.height, c0:c0 + src.width], data)

    # write final mosaic
    with rasterio.open(IMG) as src:
        meta = src.profile
    meta.update(count=1, dtype="float32", compress="deflate", nodata=0.0)

    with rasterio.open(OUT_MOSAIC, "w", **meta) as dst:
        dst.write(mosaic, 1)

    print(f"✓ final mosaic written → {OUT_MOSAIC}")

    if args.clean:
        shutil.rmtree(TMP_DIR)
        print("tmp_prob/ removed")