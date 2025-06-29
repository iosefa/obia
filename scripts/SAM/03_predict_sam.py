#!/usr/bin/env python3
"""
03_predict_sam.py  – per‑crown probability map (example: cluster 450)

Changes vs your previous revision
─────────────────────────────────
1.  **Smarter foreground sampling**
      – top‑N tallest CHM pixels inside the 12 m box (no random picks).

2.  **Gentle CHM prior**
      – only pixels higher than the seed get up to +2 logits.

3.  **Flatter Gaussian core**
      – σ = BOX_METRE / 2  (≈ 6 m) so the whole box is kept ‘alive’.

Nothing else in the pipeline is touched.
"""

from pathlib import Path
import argparse, random
import cv2, numpy as np, rasterio, torch, geopandas as gpd
from shapely.geometry import Point, box as Box
from rasterio.transform import rowcol, Affine
from segment_anything import sam_model_registry
from samgeo import SamGeo

# ─── paths ─────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
IMG    = ROOT / "image.tif"
CHM    = ROOT / "chm.tif"
SEEDS  = ROOT / "seeds_cost.gpkg"

SAM_META = ROOT / "sam_vit_l_0b3195.pth"
SAM_DEC  = ROOT / "sam_crowns_decoder.pth"

OUT_TIF  = ROOT / "seed_562_prob.tif"
OUT_GPKG = ROOT / "seed_562_box.gpkg"

# ─── parameters ────────────────────────────────────────────────────
CLUSTER_ID   = 562
BOX_METRE    = 15.0
CONTEXT_FCTR = 2.0
CHM_GROUND   = 3.0
SIGMA_METRE  = 1.5
FG_EXTRA     = 40
FG_RAD       = 2
NEG_MAX_GND  = 150
NEG_MAX_CAN  = 100
USE_CHM      = True
# ------------------------------------------------------------------

def stretch(arr, p2=2, p98=98):
    lo, hi = np.percentile(arr, (p2, p98))
    return np.clip((arr - lo) * 255 / (hi - lo + 1e-9), 0, 255).astype("uint8")

def gauss(img, σ_pix):               # replicate border to avoid dark halo
    k = int(round(σ_pix * 3)) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), σ_pix, borderType=cv2.BORDER_REPLICATE)

def resize_long(img, long=1024):
    h, w = img.shape[:2]
    s = long / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), cv2.INTER_LINEAR), s


def main(no_chm: bool):
    # 0 ── read RGB patch around the crown ─────────────────────────
    with rasterio.open(IMG) as src:
        tfm, crs, res = src.transform, src.crs, abs(src.transform.a)
        rgb16 = np.stack([src.read(i) for i in (7, 5, 3)])
        H, W = src.height, src.width

    seeds = gpd.read_file(SEEDS).to_crs(crs)
    cx, cy = seeds.loc[seeds.cluster == CLUSTER_ID,
                       "geometry"].iloc[0].coords[0]
    row, col = rowcol(tfm, cx, cy)

    half_pad = int(round(CONTEXT_FCTR * BOX_METRE / res))
    r0, r1 = max(row - half_pad, 0), min(row + half_pad, H)
    c0, c1 = max(col - half_pad, 0), min(col + half_pad, W)

    rgb_patch = np.moveaxis(rgb16[:, r0:r1, c0:c1], 0, -1)
    h_p, w_p = rgb_patch.shape[:2]

    rgb_u8 = np.stack([stretch(rgb_patch[..., i]) for i in range(3)], -1)
    rgb_1024, s = resize_long(rgb_u8)

    # 1 ── prompt geometry in *resized* pixels ─────────────────────
    half_box_px = BOX_METRE / (2 * res) * s
    rp, cp = (row - r0) * s, (col - c0) * s
    box_px = np.array([[cp - half_box_px, rp - half_box_px,
                        cp + half_box_px, rp + half_box_px]],
                      dtype="float32")

    # 2 ── foreground points ---------------------------------------
    fg = [[cp, rp]]                           # the seed itself
    with rasterio.open(CHM) as src_chm:
        chm_patch = src_chm.read(
            1, window=rasterio.windows.Window(c0, r0, w_p, h_p),
            boundless=True, fill_value=0)

    # take the FG_EXTRA tallest CHM pixels inside the box
    if FG_EXTRA:
        ys, xs = np.where(chm_patch > CHM_GROUND)
        in_box = ((abs(xs - (col - c0)) <= BOX_METRE/(FG_RAD*res)) &
                  (abs(ys - (row - r0)) <= BOX_METRE/(FG_RAD*res)))
        xs, ys = xs[in_box], ys[in_box]
        if xs.size:
            order = np.argsort(chm_patch[ys, xs])[::-1][:FG_EXTRA]
            for x, y in zip(xs[order], ys[order]):
                if x == (col - c0) and y == (row - r0):
                    continue
                fg.append([x * s, y * s])

    pt_px  = np.array(fg, dtype="float32")
    lbl_px = np.ones(len(pt_px), dtype="int32")

    # 3 ── negatives -----------------------------------------------
    ys_g, xs_g = np.where(chm_patch < CHM_GROUND)
    neg_g = np.column_stack([xs_g * s, ys_g * s]).astype("float32")
    if len(neg_g) > NEG_MAX_GND:
        neg_g = neg_g[np.random.choice(len(neg_g), NEG_MAX_GND, False)]

    ys_c, xs_c = np.where(chm_patch > CHM_GROUND)
    neg_c = []
    if NEG_MAX_CAN:
        half_box_pix = BOX_METRE / (2 * res)
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

    # 4 ── SAM ‑‑‑---------------------------------------------------
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    sam = sam_model_registry["vit_l"](checkpoint=str(SAM_META)).to(device)
    sam.load_state_dict(torch.load(SAM_DEC, map_location=device),
                        strict=False)
    sam.mask_decoder.num_multimask_outputs = 1
    sam.mask_decoder.mask_tokens = torch.nn.Embedding(
        1, sam.mask_decoder.mask_tokens.embedding_dim, device=device)

    predictor = SamGeo(model=sam, automatic=False, device=device)
    predictor.set_image(rgb_1024)

    logit_rs = predictor.predict(
        point_coords=pts,
        point_labels=lbl,
        box=box_px,
        tile=False,
        return_logits=True,
        multimask_output=False,
        return_results=True
    )[2][0]

    logit = cv2.resize(logit_rs, (w_p, h_p), interpolation=cv2.INTER_LINEAR)

    # 5 ── gentle CHM boost ----------------------------------------
    try:
        if USE_CHM and not no_chm:
            σ_pix = SIGMA_METRE / res
            chm_s = gauss(chm_patch.astype("float32"), σ_pix)

            h_seed = chm_patch[int(row - r0), int(col - c0)]
            dh_pos = np.clip(chm_s - h_seed, 0, None)          # only above seed
            logit += 2.0 * dh_pos / (dh_pos.max() + 1e-6)      # +0 … +2 logits
    except IndexError as e:
        print(e)

    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -12, 12)))

    # 6 ── smooth Gaussian emphasis -------------------------------
    yy, xx = np.indices(prob.shape)
    d_m = np.hypot(yy - (row - r0), xx - (col - c0)) * res

    σ_core = BOX_METRE / 2.0          # wider than before
    core_w = np.exp(-(d_m ** 2) / (2 * σ_core ** 2))
    prob *= core_w / core_w.max()

    # 7 ── write GeoTIFF -------------------------------------------
    meta = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": h_p, "width": w_p, "crs": crs,
        "transform": Affine(tfm.a, tfm.b, tfm.c + c0 * tfm.a,
                            tfm.d, tfm.e, tfm.f + r0 * tfm.e),
        "compress": "deflate", "nodata": None
    }
    with rasterio.open(OUT_TIF, "w", **meta) as dst:
        dst.write(prob.astype("float32"), 1)
    print("✓ probability →", OUT_TIF)

    # 8 ── QC GPKG --------------------------------------------------
    gpd.GeoDataFrame(
        {"cluster": [CLUSTER_ID, CLUSTER_ID], "type": ["box", "point"]},
        geometry=[Box(cx - BOX_METRE / 2, cy - BOX_METRE / 2,
                      cx + BOX_METRE / 2, cy + BOX_METRE / 2),
                  Point(cx, cy)],
        crs=crs
    ).to_file(OUT_GPKG, driver="GPKG")
    print("✓ prompt      →", OUT_GPKG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-chm", action="store_true",
                        help="ignore CHM weighting entirely")
    main(no_chm=parser.parse_args().no_chm)