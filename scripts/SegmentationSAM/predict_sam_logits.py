#!/usr/bin/env python3
"""
predict_sam_logits.py  –  memory‑safe per‑crown probability tiles

Layout in --out-dir:
  r<row>c<col>/glob.tif          (1 band)
  r<row>c<col>/seed_<sid>.tif    (3 bands [prob, ground, CHM])
"""

import itertools, math, shutil, sys, warnings
from pathlib import Path

import click, geopandas as gpd, numpy as np, rasterio, torch
from rasterio.windows   import Window, bounds as win_bounds
from rasterio.transform import Affine
from shapely.geometry   import box
from skimage.measure    import label
from PIL import Image
from samgeo             import SamGeo

# ─── tweakables ────────────────────────────────────────────────────
RGB_PCTL   = (2, 98)
CHM_GROUND = 3.0
NEG_GROUND = 400
MAX_POS    = 400
DISC_R     = 5
LOG_CLIP   = 12.0
SUBSAMPLE  = 3
MAX_GLOBAL_POS = 1000

# ─── helpers ───────────────────────────────────────────────────────
def stretch_u8(a16):
    lo, hi = np.percentile(a16, RGB_PCTL)
    return np.clip((a16-lo)*255/(hi-lo+1e-9), 0, 255).astype("uint8")

def sam_size(h, w):
    scale = 1024.0 / max(h, w)
    return int(math.ceil(h*scale/64))*64, int(math.ceil(w*scale/64))*64

def resize(arr, size, resample):
    return np.array(Image.fromarray(arr).resize(size, resample))

def empty_gpu(dev):
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()

# ─── CLI ───────────────────────────────────────────────────────────
@click.command(context_settings={"show_default": True})
@click.option("--image",      required=True, type=click.Path(exists=True))
@click.option("--seeds",      required=True, type=click.Path(exists=True))
@click.option("--out-dir",    required=True, type=click.Path())
@click.option("--checkpoint", "weights", required=True,
              type=click.Path(exists=True))
@click.option("--model",      default="vit_l",
              type=click.Choice(["vit_b", "vit_l", "vit_h"]))
@click.option("--device",     default="auto",
              type=click.Choice(["auto", "cpu", "cuda", "mps"]))
@click.option("--tile-size",  default=512)
@click.option("--pad",        default=128)
def main(image, seeds, out_dir, weights, model, device, tile_size, pad):

    device = ("mps"  if device == "auto" and torch.backends.mps.is_available()
         else "cuda" if device == "auto" and torch.cuda.is_available()
         else "cpu"  if device == "auto" else device)
    print("Running on", device)

    out_root = Path(out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    sam = SamGeo(model_type=model, checkpoint=weights,
                 n_extra_channels=1, automatic=False, device=device)

    with rasterio.open(image) as src:
        gdf = gpd.read_file(seeds).to_crs(src.crs)
        gdf["sid"] = np.arange(len(gdf))
        gdf = gdf[gdf.within(box(*src.bounds))]
        if gdf.empty:
            raise SystemExit("No seeds intersect raster.")

        core = tile_size
        for r0, c0 in itertools.product(range(0, src.height, core),
                                        range(0, src.width,  core)):

            core_h = min(core, src.height - r0)
            core_w = min(core, src.width  - c0)
            rp0, cp0 = max(r0-pad, 0), max(c0-pad, 0)
            rp1, cp1 = min(r0+core_h+pad, src.height), min(c0+core_w+pad, src.width)
            win  = Window(cp0, rp0, cp1-cp0, rp1-rp0)
            bbox = box(*win_bounds(win, transform=src.transform))
            seeds_t = gdf[gdf.intersects(bbox)]
            if seeds_t.empty:
                continue

            # ---- make folder for this tile -----------------------
            tile_dir = out_root / f"r{r0}c{c0}"
            if tile_dir.exists():
                shutil.rmtree(tile_dir)
            tile_dir.mkdir(parents=True, exist_ok=True)

            # ---- read 9‑band patch ------------------------------
            rgb16 = src.read([7, 5, 3], window=win).astype("float32")
            nir16 = src.read(8, window=win).astype("float32")
            chm16 = src.read(9, window=win).astype("float32")

            ground = chm16 < CHM_GROUND
            rgb8   = np.stack([stretch_u8(b) for b in rgb16])
            nir8   = stretch_u8(nir16)
            rgb8[:, ground] = 0
            nir8[ground]   = 0

            H_pad, W_pad = ground.shape
            Hs, Ws = sam_size(H_pad, W_pad)
            rgb_rs = np.stack([resize(rgb8[b], (Ws, Hs), Image.Resampling.BILINEAR)
                               for b in range(3)])
            nir_rs = resize(nir8, (Ws, Hs), Image.Resampling.BILINEAR)
            can_rs = resize((~ground).astype("uint8"),
                            (Ws, Hs), Image.Resampling.NEAREST).astype(bool)
            grd_rs = ~can_rs
            sam.set_image(np.moveaxis(rgb_rs, 0, -1))

            # ---- global canopy prior ----------------------------
            pos_rs = np.column_stack(np.where(can_rs))
            if len(pos_rs) > MAX_GLOBAL_POS:
                pos_rs = pos_rs[np.random.choice(len(pos_rs), MAX_GLOBAL_POS, False)]
            neg_rs = np.column_stack(np.where(grd_rs))
            neg_rs = neg_rs[np.random.choice(
                len(neg_rs), min(NEG_GROUND, len(neg_rs)), False)]

            pts_gl = np.vstack([pos_rs[:, ::-1], neg_rs[:, ::-1]]).astype("float32")
            lbl_gl = np.hstack([np.ones(len(pos_rs), "int32"),
                                np.zeros(len(neg_rs), "int32")])

            gl_log = sam.predict(extra_channels=nir_rs[..., None],
                                 point_coords=pts_gl, point_labels=lbl_gl,
                                 return_logits=True, multimask_output=False,
                                 return_results=True)[2][0]
            empty_gpu(device)
            gl_prob_rs = 1 / (1 + np.exp(-np.clip(gl_log, -LOG_CLIP, LOG_CLIP)))
            gl_prob_pad = resize(gl_prob_rs.astype("float32"),
                                 (W_pad, H_pad), Image.Resampling.BILINEAR)
            off_r, off_c = r0 - rp0, c0 - cp0
            gl_prob = gl_prob_pad[off_r:off_r+core_h, off_c:off_c+core_w]

            # write global file
            tr_core = (src.window_transform(win) *
                       Affine.translation(off_c, off_r))
            meta = src.profile
            meta.update(driver="GTiff", dtype="float32", nodata=None,
                        compress="deflate", count=1,
                        width=core_w, height=core_h, transform=tr_core)
            with rasterio.open(tile_dir/"glob.tif", "w", **meta) as d:
                d.write(gl_prob, 1)

            # ---- connected components for seed ROI --------------
            coarse_pad = np.zeros((H_pad, W_pad), bool)
            for geom in seeds_t.geometry:
                ry, rx = src.index(geom.x, geom.y)
                coarse_pad[ry-rp0, rx-cp0] = True
            coarse_pad |= ~ground
            coarse_lab = label(coarse_pad, connectivity=2)

            neg_rs_xy = neg_rs[:, ::-1].astype("float32")
            chm_core  = chm16[off_r:off_r+core_h, off_c:off_c+core_w]

            # ---- per‑seed tiles ---------------------------------
            for sid, geom in zip(seeds_t.sid, seeds_t.geometry):
                ry, rx = src.index(geom.x, geom.y)
                cy, cx = ry-rp0, rx-cp0
                comp_id = coarse_lab[cy, cx]

                if comp_id == 0:  # fallback small disc
                    yy, xx = np.ogrid[:H_pad, :W_pad]
                    disc = (xx-cx)**2 + (yy-cy)**2 <= DISC_R**2
                    pos_pix = np.column_stack(np.where(disc))
                else:
                    comp_mask = coarse_lab == comp_id
                    pos_pix = np.column_stack(np.where(comp_mask))[::SUBSAMPLE]

                if len(pos_pix) > MAX_POS:
                    pos_pix = pos_pix[np.random.choice(len(pos_pix),
                                                       MAX_POS, False)]

                pos_xy = pos_pix[:, ::-1].astype("float32")
                pts = np.vstack([pos_xy, neg_rs_xy])
                lbls = np.hstack([np.ones(len(pos_xy),  "int32"),
                                  np.zeros(len(neg_rs_xy), "int32")])

                # ─── ►►  NO second scaling – coordinates already in chip grid ◄◄
                # (previous pts[:,0/1] *= Ws/W_pad were removed)

                log_rs = sam.predict(extra_channels=nir_rs[..., None],
                                     point_coords=pts, point_labels=lbls,
                                     return_logits=True, multimask_output=False,
                                     return_results=True)[2][0]
                empty_gpu(device)
                prob_rs = 1 / (1 + np.exp(-np.clip(log_rs, -LOG_CLIP, LOG_CLIP)))
                prob_pad = resize(prob_rs.astype("float32"),
                                  (W_pad, H_pad), Image.Resampling.BILINEAR)
                prob_core = prob_pad[off_r:off_r+core_h, off_c:off_c+core_w]

                ground_mask = (chm_core < CHM_GROUND).astype("float32")
                cube = np.stack([prob_core, ground_mask, chm_core])

                meta.update(count=3)
                with rasterio.open(tile_dir/f"seed_{sid}.tif", "w", **meta) as d:
                    d.write(cube)

            sam.predictor.reset_image()
            empty_gpu(device)
            click.echo(f"tile r{r0} c{c0}  –  {len(seeds_t)} seeds")

    click.echo("\n✓ all probability tiles saved under", out_root)


# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        warnings.warn(str(e)); sys.exit(1)

# #!/usr/bin/env python3
# """
# predict_sam_logits.py  – memory‑safe per‑crown probability tiles
# ----------------------------------------------------------------
# Writes:
#   glob_r<row>_c<col>.tif          – 1‑band canopy prior
#   seed_<id>_r<row>_c<col>.tif     – 3 bands: [prob, ground mask, CHM]
# """
#
# import itertools, math, shutil, sys, warnings
# from pathlib import Path
# import click, geopandas as gpd, numpy as np, rasterio, torch
# from rasterio.windows   import Window, bounds as win_bounds
# from rasterio.transform import Affine
# from shapely.geometry   import box
# from skimage.measure    import label
# from PIL import Image
# from samgeo             import SamGeo
#
# # ─── tweakables ────────────────────────────────────────────────────
# RGB_PCTL   = (2, 98)
# CHM_GROUND = 3.0
# NEG_GROUND = 400
# MAX_POS    = 400          # ← new: cap pos prompts per seed
# DISC_R     = 5
# LOG_CLIP   = 12.0
# SUBSAMPLE  = 3
# MAX_GLOBAL_POS = 1000     # ← trimmed from 1500
#
# # ─── helpers -------------------------------------------------------
# def stretch_u8(a16):
#     lo, hi = np.percentile(a16, RGB_PCTL)
#     return np.clip((a16-lo)*255/(hi-lo+1e-9), 0, 255).astype("uint8")
#
# def sam_size(h, w):
#     s = 1024./max(h, w)
#     return int(math.ceil(h*s/64))*64, int(math.ceil(w*s/64))*64
#
# def resize(arr, size, resample):
#     return np.array(Image.fromarray(arr).resize(size, resample))
#
# def empty_gpu(device):
#     if device == "cuda":
#         torch.cuda.empty_cache()
#     elif device == "mps":
#         torch.mps.empty_cache()
#
# # ─── CLI -----------------------------------------------------------
# @click.command(context_settings={"show_default": True})
# @click.option("--image",      required=True, type=click.Path(exists=True))
# @click.option("--seeds",      required=True, type=click.Path(exists=True))
# @click.option("--out-dir",    required=True, type=click.Path())
# @click.option("--checkpoint", "weights", required=True,
#               type=click.Path(exists=True))
# @click.option("--model",      default="vit_l",
#               type=click.Choice(["vit_b","vit_l","vit_h"]))
# @click.option("--device",     default="auto",
#               type=click.Choice(["auto","cpu","cuda","mps"]))
# @click.option("--tile-size",  default=512)
# @click.option("--pad",        default=128)
# def main(image, seeds, out_dir, weights, model, device, tile_size, pad):
#
#     device = ( "mps"  if device=="auto" and torch.backends.mps.is_available()
#           else "cuda" if device=="auto" and torch.cuda.is_available()
#           else "cpu"  if device=="auto" else device)
#     print("Running on", device)
#
#     out = Path(out_dir);  out.mkdir(exist_ok=True, parents=True)
#
#     sam = SamGeo(model_type=model, checkpoint=weights,
#                  n_extra_channels=1, automatic=False, device=device)
#
#     with rasterio.open(image) as src:
#         gdf = gpd.read_file(seeds).to_crs(src.crs)
#         gdf["sid"] = np.arange(len(gdf))
#         gdf = gdf[gdf.within(box(*src.bounds))]
#         if gdf.empty: raise SystemExit("No seeds intersect raster")
#
#         core = tile_size
#         for r0, c0 in itertools.product(range(0, src.height, core),
#                                         range(0, src.width,  core)):
#
#             core_h = min(core, src.height - r0)
#             core_w = min(core, src.width  - c0)
#             rp0, cp0 = max(r0-pad, 0), max(c0-pad, 0)
#             rp1, cp1 = min(r0+core_h+pad, src.height), min(c0+core_w+pad, src.width)
#             win  = Window(cp0, rp0, cp1-cp0, rp1-rp0)
#             bbox = box(*win_bounds(win, transform=src.transform))
#             seeds_t = gdf[gdf.intersects(bbox)]
#             if seeds_t.empty: continue
#
#             # ---- read patch --------------------------------------
#             rgb16 = src.read([7,5,3], window=win).astype("float32")
#             nir16 = src.read(8, window=win).astype("float32")
#             chm16 = src.read(9, window=win).astype("float32")
#
#             ground = chm16 < CHM_GROUND
#             rgb8   = np.stack([stretch_u8(b) for b in rgb16])
#             nir8   = stretch_u8(nir16)
#             rgb8[:, ground] = 0;  nir8[ground] = 0
#
#             H_pad, W_pad = ground.shape
#             Hs, Ws = sam_size(H_pad, W_pad)
#             rgb_rs = np.stack([resize(rgb8[b], (Ws,Hs), Image.Resampling.BILINEAR)
#                                for b in range(3)])
#             nir_rs = resize(nir8, (Ws,Hs), Image.Resampling.BILINEAR)
#             can_rs = resize((~ground).astype("uint8"),
#                             (Ws,Hs), Image.Resampling.NEAREST).astype(bool)
#             grd_rs = ~can_rs
#             sam.set_image(np.moveaxis(rgb_rs, 0, -1))
#
#             # ---- global prior ------------------------------------
#             pos_rs = np.column_stack(np.where(can_rs))
#             if len(pos_rs) > MAX_GLOBAL_POS:
#                 pos_rs = pos_rs[np.random.choice(len(pos_rs), MAX_GLOBAL_POS, False)]
#             neg_rs = np.column_stack(np.where(grd_rs))
#             neg_rs = neg_rs[np.random.choice(len(neg_rs), min(NEG_GROUND,len(neg_rs)), False)]
#             pts_gl = np.vstack([pos_rs[:,::-1], neg_rs[:,::-1]]).astype("float32")
#             lbl_gl = np.hstack([np.ones(len(pos_rs),"int32"),
#                                 np.zeros(len(neg_rs),"int32")])
#
#             gl_log = sam.predict(extra_channels=nir_rs[...,None],
#                                  point_coords=pts_gl,
#                                  point_labels=lbl_gl,
#                                  return_logits=True,
#                                  multimask_output=False,
#                                  return_results=True)[2][0]
#             empty_gpu(device)
#             gl_prob_rs = 1/(1+np.exp(-np.clip(gl_log, -LOG_CLIP, LOG_CLIP)))
#             gl_prob_pad = resize(gl_prob_rs.astype("float32"),
#                                  (W_pad,H_pad), Image.Resampling.BILINEAR)
#             off_r, off_c = r0-rp0, c0-cp0
#             gl_prob = gl_prob_pad[off_r:off_r+core_h, off_c:off_c+core_w]
#
#             tr_core = (src.window_transform(win) * Affine.translation(off_c, off_r))
#             meta = src.profile
#             meta.update(driver="GTiff", dtype="float32", nodata=None,
#                         compress="deflate", count=1,
#                         width=core_w, height=core_h, transform=tr_core)
#             with rasterio.open(out/f"glob_r{r0}_c{c0}.tif","w",**meta) as d:
#                 d.write(gl_prob,1)
#
#             # ---- coarse components --------------------------------
#             coarse_pad = np.zeros((H_pad,W_pad), bool)
#             for geom in seeds_t.geometry:
#                 ry, rx = src.index(geom.x, geom.y)
#                 coarse_pad[ry-rp0, rx-cp0] = True
#             coarse_pad |= ~ground
#             coarse_lab = label(coarse_pad, connectivity=2)
#
#             neg_rs_xy = neg_rs[:, ::-1].astype("float32")
#             off_r, off_c = r0-rp0, c0-cp0
#             chm_c = chm16[off_r:off_r+core_h, off_c:off_c+core_w]
#
#             # ---- seed loop – write one small file per seed --------
#             for sid, geom in zip(seeds_t.sid, seeds_t.geometry):
#                 ry, rx = src.index(geom.x, geom.y)
#                 cy, cx = ry-rp0, rx-cp0
#                 comp_id = coarse_lab[cy, cx]
#
#                 if comp_id==0:
#                     yy,xx=np.ogrid[:H_pad,:W_pad]
#                     disc=(xx-cx)**2+(yy-cy)**2<=DISC_R**2
#                     pos_pix=np.column_stack(np.where(disc))
#                 else:
#                     comp_mask = coarse_lab==comp_id
#                     pos_pix = np.column_stack(np.where(comp_mask))[::SUBSAMPLE]
#
#                 if len(pos_pix) > MAX_POS:
#                     pos_pix = pos_pix[np.random.choice(len(pos_pix), MAX_POS, False)]
#
#                 pos_xy = pos_pix[:, ::-1].astype("float32")
#                 pts = np.vstack([pos_xy, neg_rs_xy])
#                 lbls = np.hstack([np.ones(len(pos_xy),"int32"),
#                                   np.zeros(len(neg_rs_xy),"int32")])
#
#                 pts[:,0] *= Ws/W_pad;  pts[:,1] *= Hs/H_pad
#
#                 log_rs = sam.predict(extra_channels=nir_rs[...,None],
#                                      point_coords=pts,
#                                      point_labels=lbls,
#                                      return_logits=True,
#                                      multimask_output=False,
#                                      return_results=True)[2][0]
#                 empty_gpu(device)
#                 prob_rs = 1/(1+np.exp(-np.clip(log_rs,-LOG_CLIP,LOG_CLIP)))
#                 prob_pad = resize(prob_rs.astype("float32"),
#                                   (W_pad,H_pad), Image.Resampling.BILINEAR)
#                 prob_core = prob_pad[off_r:off_r+core_h, off_c:off_c+core_w]
#
#                 ground_mask = (chm_c < CHM_GROUND).astype("float32")
#                 cube = np.stack([prob_core, ground_mask, chm_c])
#
#                 meta.update(count=3)
#                 fn = out/f"seed_{sid}_r{r0}_c{c0}.tif"
#                 with rasterio.open(fn,"w",**meta) as d: d.write(cube)
#
#             sam.predictor.reset_image()
#             empty_gpu(device)
#             click.echo(f"tile r{r0} c{c0}  –  {len(seeds_t)} seeds done")
#
#     click.echo("\n✓ all probability tiles saved to", out)
#
# # ------------------------------------------------------------------
# if __name__=="__main__":
#     try:
#         main()
#     except Exception as e:
#         warnings.warn(str(e)); sys.exit(1)

# #!/usr/bin/env python3
# """
# predict_sam_logits.py  –  write SAM probability cubes
# ──────────────────────────────────────────────────────
# bands per 512×512 core (pad keeps SAM context):
#
#   1        P(pixel ∈ canopy)           – global prior
#   2…K+1    P(pixel ∈ seed_k crown)     – one band per seed
#   K+2      ground mask  (CHM < 3 m →1)
#   K+3      CHM down-sampled  (m)
# """
# # -------------------------------------------------------------------
# from pathlib import Path
# import math, sys, shutil, warnings, click, itertools
# import numpy as np, geopandas as gpd, rasterio
# from rasterio.windows   import Window, bounds as win_bounds
# from rasterio.transform import Affine
# from shapely.geometry   import box
# from skimage.measure    import label
# from samgeo             import SamGeo
# from PIL import Image
# # ───────── parameters ──────────────────────────────────────────────
# RGB_PCTL   = (2, 98)
# CHM_GROUND = 3.0       # m
# NEG_GROUND = 400       # ground negatives / tile
# DISC_R     = 5         # px (≈1.5 m) fallback positive disc
# HALO_IN    = DISC_R+2
# HALO_OUT   = DISC_R+6
# LOG_CLIP   = 12.0
# SUBSAMPLE  = 3
# # helpers -----------------------------------------------------------
# def stretch_u8(a16):
#     lo, hi = np.percentile(a16, RGB_PCTL)
#     return np.clip((a16-lo)*255/(hi-lo+1e-6),0,255).astype('uint8')
# def sam_size(h,w):
#     s = 1024./max(h,w)
#     return int(math.ceil(h*s/64))*64, int(math.ceil(w*s/64))*64
# def resize(img, size, interp):
#     return np.array(Image.fromarray(img).resize(size, interp))
# # -------------------------------------------------------------------
# @click.command(context_settings={"show_default":True})
# @click.option("--image",      required=True, type=click.Path(exists=True))
# @click.option("--seeds",      required=True, type=click.Path(exists=True))
# @click.option("--out-dir",    required=True, type=click.Path())
# @click.option("--checkpoint", required=True)
# @click.option("--model",      default="vit_l",
#               type=click.Choice(["vit_b","vit_l","vit_h"]))
# @click.option("--tile-size",  default=512)
# @click.option("--pad",        default=128)
# def main(image, seeds, out_dir, checkpoint, model, tile_size, pad):
#
#     out = Path(out_dir).expanduser()
#     if out.exists(): shutil.rmtree(out)
#     out.mkdir(parents=True)
#
#     sam = SamGeo(model_type=model, checkpoint=checkpoint,
#                  n_extra_channels=1, automatic=False)
#
#     # ─── iterate tiles ────────────────────────────────────────────
#     with rasterio.open(image) as src:
#         gdf = gpd.read_file(seeds).to_crs(src.crs)
#         gdf = gdf[gdf.within(box(*src.bounds))]
#         if gdf.empty: raise SystemExit("no seeds on image")
#
#         core = tile_size
#         for r0,c0 in itertools.product(range(0,src.height,core),
#                                        range(0,src.width ,core)):
#
#             core_h = min(core, src.height-r0);  core_w = min(core, src.width-c0)
#             rp0,cp0 = max(r0-pad,0), max(c0-pad,0)
#             rp1,cp1 = min(r0+core_h+pad,src.height), min(c0+core_w+pad,src.width)
#             win     = Window(cp0,rp0,cp1-cp0,rp1-rp0)
#             bbox    = box(*win_bounds(win,transform=src.transform))
#
#             seeds_t = gdf[gdf.intersects(bbox)]
#             K = len(seeds_t)
#             if K==0: continue
#
#             # 1) read patch ------------------------------------------------
#             rgb16 = src.read([7,5,3],window=win).astype('float32')
#             nir16 = src.read(8,window=win).astype('float32')
#             chm16 = src.read(9,window=win).astype('float32')
#
#             ground = chm16<CHM_GROUND
#             rgb8   = np.stack([stretch_u8(b) for b in rgb16])
#             nir8   = stretch_u8(nir16);  rgb8[:,ground]=0;  nir8[ground]=0
#
#             H_pad,W_pad = ground.shape
#             Hs,Ws = sam_size(H_pad,W_pad)
#
#             rgb_rs = np.stack([resize(rgb8[b],(Ws,Hs),Image.Resampling.BILINEAR)
#                                for b in range(3)])
#             nir_rs = resize(nir8,(Ws,Hs),Image.Resampling.BILINEAR)
#             can_rs = resize((~ground).astype('uint8'),
#                             (Ws,Hs),Image.Resampling.NEAREST).astype(bool)
#             grd_rs = ~can_rs
#             sam.set_image(np.moveaxis(rgb_rs,0,-1))
#
#             # 2) global canopy prior ---------------------------------
#             pos_rs = np.column_stack(np.where(can_rs))
#             if len(pos_rs)>1500: pos_rs=pos_rs[np.random.choice(len(pos_rs),1500,False)]
#             neg_rs = np.column_stack(np.where(grd_rs))
#             neg_rs = neg_rs[np.random.choice(len(neg_rs),
#                                              min(NEG_GROUND,len(neg_rs)),False)]
#             pts_gl = np.vstack([pos_rs[:,::-1], neg_rs[:,::-1]]).astype('float32')
#             lbl_gl = np.hstack([np.ones(len(pos_rs),'int32'),
#                                 np.zeros(len(neg_rs),'int32')])
#             gl_log = sam.predict(extra_channels=nir_rs[...,None],
#                                  point_coords=pts_gl, point_labels=lbl_gl,
#                                  return_logits=True, multimask_output=False,
#                                  return_results=True)[2][0]
#             gl_prob_rs = 1/(1+np.exp(-np.clip(gl_log,-LOG_CLIP,LOG_CLIP)))
#
#             # 3) coarse mask on **padded grid** ----------------------
#             coarse_pad = np.zeros((H_pad,W_pad),bool)
#             for geom in seeds_t.geometry:
#                 ry,rx = src.index(geom.x,geom.y)
#                 coarse_pad[ry-rp0, rx-cp0]=True
#             coarse_pad = np.maximum(coarse_pad, ~ground)   # include canopy pixels
#             coarse_lab = label(coarse_pad, connectivity=2)
#
#             # 4) per-seed refinement --------------------------------
#             probs = np.empty((K,core_h,core_w),'float32')
#             for k,geom in enumerate(seeds_t.geometry):
#                 ry,rx = src.index(geom.x,geom.y)
#                 cy,cx = ry-rp0, rx-cp0
#                 comp_id = coarse_lab[cy,cx]
#                 if comp_id==0:                            # fallback disc
#                     yy,xx = np.ogrid[:H_pad,:W_pad]
#                     disc = (xx-cx)**2 + (yy-cy)**2 <= DISC_R**2
#                     pos_pix = np.column_stack(np.where(disc))
#                 else:
#                     comp_mask = coarse_lab==comp_id
#                     pos_pix = np.column_stack(np.where(comp_mask))[::SUBSAMPLE]
#
#                 pos_xy = pos_pix[:,::-1].astype('float32')
#                 pts  = np.vstack([pos_xy, neg_rs[:,::-1].astype('float32')])
#                 lbls = np.hstack([np.ones(len(pos_xy),'int32'),
#                                   np.zeros(len(neg_rs),'int32')])
#
#                 # map pos_xy from padded to RS grid
#                 pts[:,0] = pts[:,0]*Ws/W_pad
#                 pts[:,1] = pts[:,1]*Hs/H_pad
#                 pts = np.clip(pts,0,max(Ws-1,Hs-1))
#
#                 log_rs = sam.predict(extra_channels=nir_rs[...,None],
#                                      point_coords=pts,
#                                      point_labels=lbls,
#                                      return_logits=True,
#                                      multimask_output=False,
#                                      return_results=True)[2][0]
#                 prob_rs = 1/(1+np.exp(-np.clip(log_rs,-LOG_CLIP,LOG_CLIP)))
#                 prob_pad = resize(prob_rs.astype('float32'),
#                                   (W_pad,H_pad),Image.Resampling.BILINEAR)
#                 pt_off,pl_off = r0-rp0, c0-cp0
#                 probs[k]=prob_pad[pt_off:pt_off+core_h, pl_off:pl_off+core_w]
#
#             # 5) crop global prior & CHM -----------------------------
#             gl_prob_pad = resize(gl_prob_rs.astype('float32'),
#                                  (W_pad,H_pad),Image.Resampling.BILINEAR)
#             pt_off,pl_off = r0-rp0, c0-cp0
#             gl_prob = gl_prob_pad[pt_off:pt_off+core_h, pl_off:pl_off+core_w]
#             chm_c   = chm16   [pt_off:pt_off+core_h, pl_off:pl_off+core_w]
#
#             H_log,W_log = probs.shape[1:]
#             rmap=(np.arange(H_log)*core_h/H_log).astype(int)
#             cmap=(np.arange(W_log)*core_w/W_log).astype(int)
#             chm_ds = chm_c[np.ix_(rmap,cmap)]
#             mask_ds= (chm_ds<CHM_GROUND).astype('float32')
#
#             cube=np.concatenate([gl_prob[None,...], probs,
#                                  mask_ds[None,...], chm_ds[None,...]],0)
#
#             tr_core=(src.window_transform(win)*Affine.translation(pl_off,pt_off))
#             meta=src.profile
#             meta.update(driver='GTiff',dtype='float32',nodata=None,
#                         compress='deflate',count=cube.shape[0],
#                         width=core_w,height=core_h,transform=tr_core)
#
#             fn=out/f"sam_logits_r{r0}_c{c0}.tif"
#             with rasterio.open(fn,'w',**meta) as dst: dst.write(cube)
#             click.echo(f"• {fn.name:<28} crowns: {K}")
#
#     click.echo("\n✓ tiles written →", out)
#
# # -------------------------------------------------------------------
# if __name__=="__main__":
#     try:
#         main()
#     except Exception as e:
#         warnings.warn(str(e)); sys.exit(1)
#
# # #!/usr/bin/env python3
# # """
# # predict_sam_logits.py   – tile‑wise SAM probability cubes
# # ──────────────────────────────────────────────────────────
# # GeoTIFF bands per 512×512 “core” (pad keeps SAM context):
# #
# # 1 … K   : P(pixel ∈ seed_k)   – soft‑max over all K seeds in the tile
# # K + 1   : ground mask         – CHM < 3 m  → 1
# # K + 2   : CHM (m)             – down‑sampled to logits grid
# # """
# # from pathlib import Path
# # import math, sys, shutil, warnings, click
# # import numpy as np, geopandas as gpd, rasterio
# # from rasterio.windows import bounds as win_bounds
# # from rasterio.windows import Window
# # from rasterio.transform import Affine
# # from shapely.geometry import box
# # from samgeo import SamGeo
# #
# # RGB_PCTL   = (2, 98)
# # CHM_GROUND = 3.0          # metres
# #
# # # ───────────────────────── helpers ───────────────────────────────────
# # def stretch_u8(arr16, pct=(2, 98)):
# #     lo, hi = np.percentile(arr16, pct)
# #     return np.clip((arr16 - lo) * 255.0 / (hi - lo + 1e-6), 0, 255).astype("uint8")
# #
# # def ground_mask(chm, h_thr=CHM_GROUND):
# #     return (chm < h_thr).astype("float32")
# #
# # # ───────────────────────── CLI ───────────────────────────────────────
# # @click.command()
# # @click.option("--image",   type=click.Path(exists=True), required=True)
# # @click.option("--seeds",   type=click.Path(exists=True), required=True)
# # @click.option("--out-dir", type=click.Path(),            required=True)
# # @click.option("--checkpoint", required=True)
# # @click.option("--model",  default="vit_l",
# #               type=click.Choice(["vit_b", "vit_l", "vit_h"]))
# # @click.option("--tile-size", default=512, show_default=True)
# # @click.option("--pad",       default=128, show_default=True)
# # def main(image, seeds, out_dir, checkpoint, model, tile_size, pad):
# #
# #     out_dir = Path(out_dir)
# #     if out_dir.exists():
# #         shutil.rmtree(out_dir)
# #     out_dir.mkdir(parents=True, exist_ok=True)
# #
# #     sam = SamGeo(model_type=model, checkpoint=checkpoint,
# #                  n_extra_channels=1, automatic=False)
# #
# #     # ----------------------------------------------------------------
# #     with rasterio.open(image) as src:
# #         gdf = gpd.read_file(seeds).to_crs(src.crs)
# #         gdf = gdf[gdf.within(box(*src.bounds))]
# #         if gdf.empty:
# #             raise SystemExit("no seeds overlap the image")
# #
# #         seed_xy = gdf.geometry.apply(lambda p: (p.x, p.y)).to_numpy()
# #
# #         core = tile_size
# #         n_rows = math.ceil(src.height / core)
# #         n_cols = math.ceil(src.width  / core)
# #
# #         for ty in range(n_rows):
# #             for tx in range(n_cols):
# #                 r0, c0   = ty * core, tx * core
# #                 core_h   = min(core, src.height - r0)
# #                 core_w   = min(core, src.width  - c0)
# #
# #                 # padded window (row/col)
# #                 rp0, cp0 = max(r0 - pad, 0), max(c0 - pad, 0)
# #                 rp1, cp1 = min(r0 + core_h + pad, src.height), \
# #                            min(c0 + core_w + pad, src.width)
# #                 win = Window(cp0, rp0, cp1 - cp0, rp1 - rp0)
# #
# #                 # bbox of padded window in map coords
# #                 # minx, maxy = src.xy(rp1, cp0)         # lower‑left
# #                 # maxx, miny = src.xy(rp0, cp1)         # upper‑right
# #                 # bbox_pad   = box(minx, miny, maxx, maxy)
# #                 bbox_pad = box(*win_bounds(win, transform=src.transform))
# #
# #                 # seeds inside padded window
# #                 seeds_tile = gdf[gdf.intersects(bbox_pad)]
# #                 K = len(seeds_tile)
# #                 if K == 0:
# #                     continue
# #
# #                 # read bands in padded window
# #                 rgb16 = src.read([7, 5, 2], window=win).astype("float32")
# #                 nir16 = src.read(9,        window=win).astype("float32")
# #                 chm16 = src.read(9,        window=win).astype("float32")
# #
# #                 rgb8 = np.stack([stretch_u8(b, RGB_PCTL) for b in rgb16], 0)
# #                 nir8 = stretch_u8(nir16, RGB_PCTL)
# #
# #                 ground = chm16 < CHM_GROUND
# #                 rgb8[:, ground] = 0
# #                 nir8[ground]    = 0
# #
# #                 sam.set_image(np.moveaxis(rgb8, 0, -1))
# #
# #                 # ─── GET ONE LOGIT PLANE PER SEED ──────────────────
# #                 logits_list = []
# #                 for pt in seeds_tile.geometry:  # <- pt is a shapely Point
# #                     x, y = pt.x, pt.y
# #                     # pixel coords inside padded window
# #                     row, col = src.index(x, y)
# #                     xy_pad = np.array([[col - cp0, row - rp0]], dtype="float32")
# #                     logit = sam.predict(extra_channels=nir8[..., None],
# #                                         point_coords=xy_pad,
# #                                         point_labels=[1],
# #                                         return_logits=True,
# #                                         multimask_output=False,
# #                                         return_results=True)[2][0]  # (H,W)
# #                     logits_list.append(logit.astype("float32"))
# #
# #                 logits = np.stack(logits_list, 0)          # (K,H,W)
# #
# #                 # soft‑max along K to obtain probabilities
# #                 logits -= logits.max(0, keepdims=True)
# #                 probs = np.exp(logits)
# #                 probs /= probs.sum(0, keepdims=True)       # (K,H,W)
# #
# #                 # crop pad → core
# #                 pad_t, pad_l = r0 - rp0, c0 - cp0
# #                 probs = probs[:, pad_t:pad_t+core_h, pad_l:pad_l+core_w]
# #
# #                 # background & CHM
# #                 H_log, W_log = probs.shape[1:]
# #                 rr = (np.arange(H_log) * core_h / H_log).astype(int)
# #                 cc = (np.arange(W_log) * core_w / W_log).astype(int)
# #                 chm_core = chm16[pad_t:pad_t+core_h, pad_l:pad_l+core_w]
# #                 chm_ds   = chm_core[np.ix_(rr, cc)]
# #                 mask_ds  = ground_mask(chm_ds)
# #
# #                 cube = np.concatenate([probs, mask_ds[None,...], chm_ds[None,...]], 0)
# #
# #                 # geo‑transform for core (no pad)
# #                 tr_core = (src.window_transform(win) *
# #                            Affine.translation(pad_l, pad_t))
# #                 meta = src.meta
# #                 meta.update(driver="GTiff", dtype="float32", nodata=None,
# #                             compress="deflate",
# #                             count=cube.shape[0], width=core_w, height=core_h,
# #                             transform=tr_core)
# #
# #                 out = out_dir / f"sam_logits_r{r0}_c{c0}.tif"
# #                 with rasterio.open(out, "w", **meta) as dst:
# #                     dst.write(cube)
# #                 click.echo(f"• {out.name:<28} seed bands: {K}")
# #
# #     click.echo("\n✓ all tiles written →", out_dir)
# #
# # # ──────────────────────────────────────────────────────────────────────
# # if __name__ == "__main__":
# #     try:
# #         main()
# #     except Exception as exc:
# #         warnings.warn(str(exc))
# #         sys.exit(1)
#
# # #!/usr/bin/env python3
# # """
# # Write one GeoTIFF per tile:
# #
# #   • bands 1 … N   : raw SAM logits  (one per canonical seed inside the tile)
# #   • band  N + 1   : binary ground mask  (CHM < 3 m → 1, else 0)
# #   • band  N + 2   : CHM resampled to logits grid  (continuous background)
# #
# # Everything is tiled so that each output GeoTIFF is self‑contained and
# # perfectly aligned for the CRF stage.
# # """
# # # --------------------------------------------------------------------
# # from pathlib import Path
# # import sys, math, shutil, click, geopandas as gpd
# # import numpy as np, rasterio
# # from rasterio.windows import Window
# # from rasterio.transform import Affine
# # from shapely.geometry import box
# # from samgeo import SamGeo
# #
# # RGB_PCTL   = (2, 98)
# # CHM_GROUND = 3.0
# #
# #
# # # --------------------------------------------------------------------
# # @click.command()
# # @click.option("--image",      type=click.Path(exists=True), required=True,
# #               help="9‑band WV‑3 stack  (bands: R,G,B = 5,3,2  | NIR = 7  | CHM = 9)")
# # @click.option("--seeds",      type=click.Path(exists=True), required=True,
# #               help="Canonical seed points (any vector format supported by GeoPandas)")
# # @click.option("--out-dir",    type=click.Path(),            required=True)
# # @click.option("--model",      default="vit_b",
# #               type=click.Choice(["vit_b", "vit_l", "vit_h"]))
# # @click.option("--checkpoint", required=True, help="Path or URL to SAM .pth checkpoint")
# # @click.option("--tile-size",  default=512, show_default=True)
# # @click.option("--pad",        default=128, show_default=True)
# # def main(image, seeds, out_dir, model, checkpoint, tile_size, pad):
# #     out_dir = Path(out_dir)
# #     if out_dir.exists():
# #         shutil.rmtree(out_dir)
# #     out_dir.mkdir(parents=True, exist_ok=True)
# #
# #     # ── SAM initialisation (runs on GPU if available) ───────────────
# #     sam = SamGeo(model_type=model, checkpoint=checkpoint,
# #                  n_extra_channels=1,  # one NIR band
# #                  automatic=False)
# #
# #     # ── read all canonical seeds once ───────────────────────────────
# #     seeds_gdf = gpd.read_file(seeds)
# #
# #     # ── iterate over tiles ──────────────────────────────────────────
# #     with rasterio.open(image) as src:
# #         # keep only seeds inside the entire image
# #         seeds_in = seeds_gdf[seeds_gdf.within(box(*src.bounds))].copy()
# #         if seeds_in.empty:
# #             raise SystemExit("No seeds overlap the image extent.")
# #
# #         # cache (row, col) indices in the 9‑band image + world coords
# #         seeds_rc = [(src.index(pt.x, pt.y)[::-1] + pt.coords[0])
# #                     for pt in seeds_in.geometry]
# #
# #         core_sz = max(1, min(tile_size, 1024 - 2*pad))
# #         n_rows  = math.ceil(src.height / core_sz)
# #         n_cols  = math.ceil(src.width  / core_sz)
# #
# #         tile_ct = 0
# #         for ty in range(n_rows):
# #             for tx in range(n_cols):
# #                 row0, col0 = ty * core_sz, tx * core_sz
# #                 core_h     = min(core_sz, src.height - row0)
# #                 core_w     = min(core_sz, src.width  - col0)
# #
# #                 # padded window (keeps SAM context)
# #                 ro = max(row0 - pad, 0); co = max(col0 - pad, 0)
# #                 rh = min(row0 + core_h + pad, src.height)
# #                 cw = min(col0 + core_w + pad, src.width)
# #                 win = Window(co, ro, cw - co, rh - ro)
# #
# #                 # seeds whose *point* lies in the core area (exclude pad)
# #                 core_bounds = box(*src.xy(row0, col0, offset="ul"),
# #                                   *src.xy(row0 + core_h, col0 + core_w, offset="lr"))
# #                 coords_tile = [[c - co, r - ro]
# #                                for (r, c, _, _), pt
# #                                in zip(seeds_rc, seeds_in.geometry)
# #                                if pt.within(core_bounds)]
# #
# #                 if not coords_tile:           # no seeds in this tile
# #                     continue
# #
# #                 # ── read image bands ──────────────────────────────
# #                 rgb16 = src.read([7, 5, 3], window=win).astype("float32")  # (3,h,w)
# #                 nir16 = src.read(8,        window=win).astype("float32")   # (h,w)
# #                 chm16 = src.read(9,        window=win).astype("float32")   # (h,w)
# #
# #                 # ground mask at full res
# #                 ground_native = chm16 < CHM_GROUND
# #
# #                 # uint‑8 stretch for SAM (simple per‑band percentile)
# #                 rgb8 = np.empty_like(rgb16, dtype=np.uint8)
# #                 for i in range(3):
# #                     lo, hi = np.percentile(rgb16[i], RGB_PCTL)
# #                     rgb8[i] = np.clip((rgb16[i] - lo) * 255 / (hi - lo + 1e-6), 0, 255)
# #                 nir8 = np.clip((nir16 - np.percentile(nir16, RGB_PCTL[0])) * 255 /
# #                                (np.percentile(nir16, RGB_PCTL[1]) -
# #                                 np.percentile(nir16, RGB_PCTL[0]) + 1e-6),
# #                                0, 255).astype(np.uint8)
# #
# #                 # zero out ground pixels
# #                 rgb8[:, ground_native] = 0
# #                 nir8[ ground_native]   = 0
# #
# #                 sam.set_image(np.moveaxis(rgb8, 0, -1))  # (H,W,3)
# #
# #                 logit_bands = []  # ← collect here
# #                 for xy in coords_tile:  # loop over seeds
# #                     xy_arr = np.asarray(xy, dtype=np.float32)[None, :]  # shape (1,2)
# #                     logits = sam.predict(extra_channels=nir8[..., None],
# #                                          point_coords=xy_arr,
# #                                          point_labels=[1],  # foreground
# #                                          return_logits=True,
# #                                          multimask_output=False,
# #                                          return_results=True)[2]  # (1,H,W)
# #                     # logits returned as (1,H,W); keep the plane
# #                     logit_bands.append(logits[0])
# #
# #                 # stack → (N_seeds, h_out, w_out)
# #                 logits = np.stack(logit_bands, 0)
# #                 # SAM inference – **one call, many points**
# #                 # sam.set_image(np.moveaxis(rgb8, 0, -1))         # (H,W,3)
# #                 #
# #                 # masks, scores, logits_list = sam.predict(
# #                 #     extra_channels   = nir8[..., None],          # (H,W,1)
# #                 #     point_coords     = coords_tile,
# #                 #     point_labels     = [1] * len(coords_tile),
# #                 #     return_logits    = True,
# #                 #     multimask_output = False,
# #                 #     return_results   = True
# #                 # )
# #                 #
# #                 # # `logits_list` is a list length = # points; each item shape (1,h,w)
# #                 # logits = np.stack([l[0] if l.ndim == 3 else l
# #                 #                    for l in logits_list], 0)     # (N_seeds,h,w)
# #
# #                 # downsample CHM & ground mask to logits grid (nearest)
# #                 h_out, w_out = logits.shape[1:]
# #                 row_idx = (np.arange(h_out) * chm16.shape[0] / h_out).astype(int)
# #                 col_idx = (np.arange(w_out) * chm16.shape[1] / w_out).astype(int)
# #                 chm_ds  = chm16[np.ix_(row_idx, col_idx)]
# #                 mask_ds = (chm_ds < CHM_GROUND).astype(np.float32)
# #
# #                 # final stack: [ logits … , ground , CHM ]
# #                 stack = np.concatenate([logits,
# #                                         mask_ds[None, ...],
# #                                         chm_ds[None,  ...]], 0)
# #
# #                 # precise transform for the down‑sampled grid
# #                 tr = src.window_transform(win) * Affine.scale(
# #                         chm16.shape[1] / w_out,      # pixel width   scale
# #                         chm16.shape[0] / h_out)      # pixel height  scale
# #
# #                 profile = src.profile
# #                 profile.update(driver="GTiff", compress="deflate",
# #                                dtype="float32", nodata=None,
# #                                count = stack.shape[0],
# #                                width = w_out, height = h_out,
# #                                transform = tr)
# #
# #                 out = out_dir / f"sam_logits_r{row0}_c{col0}.tif"
# #                 with rasterio.open(out, "w", **profile) as dst:
# #                     dst.write(stack.astype("float32"))
# #
# #                 click.echo(f" • {out.name:<35}  seeds: {logits.shape[0]}")
# #                 tile_ct += 1
# #
# #     click.echo(f"✓ {tile_ct} logit tiles written → {out_dir}")
# #
# #
# # # --------------------------------------------------------------------
# # if __name__ == "__main__":
# #     try:
# #         main()
# #     except Exception as exc:
# #         click.echo(f"Error: {exc}", err=True)
# #         sys.exit(1)
#
# # #!/usr/bin/env python3
# # """
# # Write one GeoTIFF per tile:
# #   • bands 1 … N  : raw SAM logits (one per canonical seed)
# #   • band N + 1   : binary ground mask  (CHM < 3 m → 1)
# #   • band N + 2   : CHM resampled to logits grid  (continuous background cue)
# # """
# # from pathlib import Path
# # import sys, shutil, math
# # import click, geopandas as gpd, numpy as np, rasterio
# # from rasterio.windows import Window
# # from rasterio.transform import Affine
# # from shapely.geometry import box
# # from samgeo import SamGeo
# #
# # RGB_PCTL, CHM_GROUND = (2, 98), 3.0
# #
# #
# # @click.command()
# # @click.option("--image",      type=click.Path(exists=True), required=True)
# # @click.option("--seeds",      type=click.Path(exists=True), required=True)
# # @click.option("--out-dir",    type=click.Path(),            required=True)
# # @click.option("--model",      default="vit_b",
# #               type=click.Choice(["vit_b", "vit_l", "vit_h"]))
# # @click.option("--checkpoint", required=True)
# # @click.option("--tile-size",  default=512, show_default=True)
# # @click.option("--pad",        default=128, show_default=True)
# # def main(image, seeds, out_dir, model, checkpoint, tile_size, pad):
# #     out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
# #     sam = SamGeo(model_type=model, checkpoint=checkpoint,
# #                  n_extra_channels=1, automatic=False)
# #
# #     seeds_gdf = gpd.read_file(seeds)
# #     with rasterio.open(image) as src:
# #         seeds_in = seeds_gdf[seeds_gdf.within(box(*src.bounds))].copy()
# #         if seeds_in.empty:
# #             raise SystemExit("No seeds overlap image.")
# #         # cache (row,col) & world coords
# #         seeds_rc = [(src.index(pt.x, pt.y)[::-1] + pt.coords[0])
# #                     for pt in seeds_in.geometry]
# #
# #         core = max(1, min(tile_size, 1024 - 2*pad))
# #         for row0 in range(0, src.height, core):
# #             for col0 in range(0, src.width, core):
# #                 core_h = min(core, src.height-row0)
# #                 core_w = min(core, src.width -col0)
# #
# #                 # padded window
# #                 ro = max(row0-pad, 0); co = max(col0-pad, 0)
# #                 rh = min(row0+core_h+pad, src.height)
# #                 cw = min(col0+core_w+pad, src.width)
# #                 win = Window(co, ro, cw-co, rh-ro)
# #
# #                 # seeds inside the core (exclude pad)
# #                 core_bounds = box(*src.xy(row0, col0, offset="ul"),
# #                                   *src.xy(row0+core_h, col0+core_w, offset="lr"))
# #                 coords = [[c-co, r-ro]
# #                           for (r,c,_,_), pt in zip(seeds_rc, seeds_in.geometry)
# #                           if pt.within(core_bounds)]
# #                 if not coords:    # no seeds here
# #                     continue
# #
# #                 # read bands
# #                 rgb16 = src.read([7,5,3], window=win).astype("float32")
# #                 nir16 = src.read(8,     window=win).astype("float32")
# #                 chm16 = src.read(9,     window=win).astype("float32")
# #
# #                 ground_native = chm16 < CHM_GROUND
# #
# #                 # uint8 stretch
# #                 rgb8 = np.empty_like(rgb16, dtype=np.uint8)
# #                 for i in range(3):
# #                     lo, hi = np.percentile(rgb16[i], RGB_PCTL)
# #                     rgb8[i] = np.clip((rgb16[i]-lo)*255/(hi-lo+1e-6), 0, 255)
# #                 nir8 = np.clip((nir16 - np.percentile(nir16, RGB_PCTL[0]))*255 /
# #                                (np.percentile(nir16, RGB_PCTL[1]) -
# #                                 np.percentile(nir16, RGB_PCTL[0]) + 1e-6),
# #                                0, 255).astype(np.uint8)
# #
# #                 rgb8[:, ground_native] = 0
# #                 nir8[ ground_native]   = 0
# #
# #                 sam.set_image(np.moveaxis(rgb8, 0, -1))
# #                 logits = sam.predict(
# #                             extra_channels=nir8[..., None],
# #                             point_coords=coords,
# #                             point_labels=[1]*len(coords),
# #                             return_logits=True,
# #                             multimask_output=False,
# #                             return_results=True
# #                          )[2]                                  # (N,h,w)
# #
# #                 # down‑sample CHM & mask to logits grid
# #                 h_out, w_out = logits.shape[1:]
# #                 row_idx = (np.arange(h_out) * chm16.shape[0] / h_out).astype(int)
# #                 col_idx = (np.arange(w_out) * chm16.shape[1] / w_out).astype(int)
# #                 chm_ds = chm16[np.ix_(row_idx, col_idx)]
# #                 mask_ds = (chm_ds < CHM_GROUND).astype(np.float32)
# #
# #                 stack = np.concatenate([logits,
# #                                         mask_ds[None, ...],
# #                                         chm_ds[None, ...]], 0)
# #
# #                 tr = src.window_transform(win) * Affine.scale(
# #                         chm16.shape[1]/w_out, chm16.shape[0]/h_out)
# #
# #                 prof = src.profile
# #                 prof.update(driver="GTiff", compress="deflate", dtype="float32",
# #                             nodata=None, count=stack.shape[0],
# #                             width=w_out, height=h_out, transform=tr)
# #
# #                 out = out_dir / f"sam_logits_r{row0}_c{col0}.tif"
# #                 with rasterio.open(out, "w", **prof) as dst:
# #                     dst.write(stack)
# #     click.echo("✓ logits written →", out_dir)
# #
# #
# # if __name__ == "__main__":
# #     try:
# #         main()
# #     except Exception as exc:
# #         click.echo(f"Error: {exc}", err=True)
# #         sys.exit(1)
#
# # #!/usr/bin/env python3
# # """
# # predict_sam_logits.py  (REV-12 • keep pad, align CHM)
# #
# # Output per tile:
# #   • Bands 1 … N      : SAM logits
# #   • Band  N + 1      : ground mask  (1 where CHM < 3 m, else 0)
# #   • Band  N + 2      : CHM down-sampled to logits grid
# # """
# # from pathlib import Path
# # import sys, shutil, math
# # import click, geopandas as gpd, numpy as np, rasterio
# # from rasterio.windows import Window
# # from rasterio.transform import Affine
# # from shapely.geometry import box
# # from samgeo import SamGeo
# #
# # RGB_PCTL   = (2, 98)
# # CHM_GROUND = 3.0
# #
# # # ─── CLI ───────────────────────────────────────────────────────────────
# # @click.command()
# # @click.option('--image',      type=click.Path(exists=True), required=True,
# #               help='9-band WV-3 stack: RGB=5/3/2, NIR=7, CHM=9')
# # @click.option('--seeds',      type=click.Path(exists=True), required=True)
# # @click.option('--out-dir',    type=click.Path(),            required=True)
# # @click.option('--model',      default='vit_b',
# #               type=click.Choice(['vit_b', 'vit_l', 'vit_h']))
# # @click.option('--checkpoint', required=True,
# #               help='Path or URL to SAM .pth checkpoint')
# # @click.option('--tile-size',  default=512, show_default=True)
# # @click.option('--pad',        default=128, show_default=True)
# # def main(image, seeds, out_dir, model, checkpoint, tile_size, pad):
# #     out_dir = Path(out_dir)
# #     if out_dir.exists():
# #         shutil.rmtree(out_dir)
# #     out_dir.mkdir(parents=True, exist_ok=True)
# #
# #     seeds_gdf = gpd.read_file(seeds)
# #
# #     with rasterio.open(image) as src:
# #         seeds_in = seeds_gdf[seeds_gdf.within(box(*src.bounds))].copy()
# #         if seeds_in.empty:
# #             raise SystemExit('No seeds overlap image extent.')
# #         click.echo(f'• seeds fed to SAM: {len(seeds_in):,}')
# #
# #         seeds_rc = [(src.index(pt.x, pt.y)[::-1] + pt.coords[0])
# #                     for pt in seeds_in.geometry]
# #
# #         core_sz = max(1, min(tile_size, 1024 - 2*pad))
# #         n_rows  = math.ceil(src.height / core_sz)
# #         n_cols  = math.ceil(src.width  / core_sz)
# #
# #         sam = SamGeo(model_type=model,
# #                      checkpoint=checkpoint,
# #                      n_extra_channels=1,
# #                      automatic=False)
# #
# #         tile_ct = 0
# #         for ty in range(n_rows):
# #             for tx in range(n_cols):
# #                 row0, col0 = ty * core_sz, tx * core_sz
# #                 core_h = min(core_sz, src.height - row0)
# #                 core_w = min(core_sz, src.width  - col0)
# #
# #                 # padded window
# #                 row_off = max(row0 - pad, 0); col_off = max(col0 - pad, 0)
# #                 row_end = min(row0 + core_h + pad, src.height)
# #                 col_end = min(col0 + core_w + pad, src.width)
# #                 window  = Window(col_off, row_off,
# #                                  col_end - col_off, row_end - row_off)
# #
# #                 # seeds inside core (not pad)
# #                 core_bounds = box(*src.xy(row0, col0, offset='ul'),
# #                                   *src.xy(row0 + core_h, col0 + core_w, offset='lr'))
# #                 coords_tile = [[c - col_off, r - row_off]
# #                                for (r, c, _, _), pt
# #                                in zip(seeds_rc, seeds_in.geometry)
# #                                if pt.within(core_bounds)]
# #                 if not coords_tile:
# #                     continue
# #
# #                 # ─── read bands ─────────────────────────────────────────
# #                 rgb16 = src.read([7, 5, 3], window=window).astype('float32')
# #                 nir16 = src.read(8,        window=window).astype('float32')
# #                 chm16 = src.read(9,        window=window).astype('float32')
# #
# #                 # native-res ground mask
# #                 ground_native = chm16 < CHM_GROUND
# #
# #                 # uint8 stretch for SAM
# #                 rgb8 = np.empty_like(rgb16, dtype=np.uint8)
# #                 for i in range(3):
# #                     lo, hi = np.percentile(rgb16[i], RGB_PCTL)
# #                     rgb8[i] = np.clip((rgb16[i] - lo) * 255 / (hi - lo + 1e-6), 0, 255)
# #                 nir8 = np.clip((nir16 - np.percentile(nir16, RGB_PCTL[0])) * 255 /
# #                                (np.percentile(nir16, RGB_PCTL[1]) -
# #                                 np.percentile(nir16, RGB_PCTL[0]) + 1e-6),
# #                                0, 255).astype(np.uint8)
# #
# #                 rgb8[:, ground_native] = 0
# #                 nir8[ ground_native]   = 0
# #
# #                 sam.set_image(np.moveaxis(rgb8, 0, -1))         # (H,W,3)
# #                 logits = sam.predict(
# #                             extra_channels=nir8[..., None],      # (H,W,1)
# #                             point_coords=coords_tile,
# #                             point_labels=[1]*len(coords_tile),
# #                             return_logits=True,
# #                             multimask_output=False,
# #                             return_results=True
# #                          )[2]                                    # (N,h_out,w_out)
# #
# #                 # ─── DOWN-SAMPLE  CHM  +  MASK  to logits grid ─────────
# #                 h_out, w_out = logits.shape[1:]
# #                 row_ratio = chm16.shape[0] / h_out
# #                 col_ratio = chm16.shape[1] / w_out
# #
# #                 rows = (np.floor(np.arange(h_out) * row_ratio)).astype(int)
# #                 cols = (np.floor(np.arange(w_out) * col_ratio)).astype(int)
# #                 chm_ds = chm16[np.ix_(rows, cols)]
# #                 bg_band = (chm_ds < CHM_GROUND).astype(np.float32)
# #
# #                 # final stack
# #                 stack = np.concatenate([logits,
# #                                         bg_band[None, ...],
# #                                         chm_ds[None, ...]], axis=0)
# #
# #                 # ─── WRITE tile ─────────────────────────────────────────
# #                 transform = (src.window_transform(window)
# #                              * Affine.scale(col_ratio, row_ratio))
# #
# #                 profile = src.profile
# #                 profile.update(driver='GTiff',
# #                                compress='deflate',
# #                                dtype='float32',
# #                                nodata=None,
# #                                count=stack.shape[0],
# #                                width=w_out,
# #                                height=h_out,
# #                                transform=transform)
# #
# #                 out = out_dir / f'sam_logits_r{row0}_c{col0}.tif'
# #                 with rasterio.open(out, 'w', **profile) as dst:
# #                     dst.write(stack.astype('float32'))
# #                 click.echo(f'  • wrote {out.name}')
# #                 tile_ct += 1
# #
# #         click.echo(f'✓ {tile_ct} logit tiles written → {out_dir}')
# #
# # if __name__ == '__main__':
# #     try:
# #         main()
# #     except Exception as exc:
# #         click.echo(f'Error: {exc}', err=True)
# #         sys.exit(1)
#
#
# # #!/usr/bin/env python3
# # """
# # predict_sam_logits.py  (REV-8 · RGB + NIR extra, ground-masked, GeoTIFF)
# # ────────────────────────────────────────────────────────────────────────
# # * **RGB**  = WV-3 bands 5 / 3 / 2  (Red, Green, Blue)
# # * **NIR**  = band 7  → passed via extra_channels (n_extra_channels=1)
# # * **CHM**  = band 9  → *only* for masking ground (pixels < 1 m → 0)
# # * Outputs one float-32 GeoTIFF of logits per tile (no seed GPKGs).
# # """
# # from pathlib import Path
# # import sys, shutil, math
# # import click, geopandas as gpd, numpy as np, rasterio
# # from rasterio.windows import Window
# # from rasterio.transform import Affine
# # from shapely.geometry import box, Point
# # from samgeo import SamGeo
# #
# # RGB_PCTL   = (2, 98)   # percentile stretch for uint8 conversion
# # CHM_GROUND = 3.0       # metres; below this is ground / soil
# #
# # # ───────────── CLI ─────────────
# # @click.command()
# # @click.option('--image',      type=click.Path(exists=True), required=True,
# #               help='9-band WV-3 stack: RGB=5/3/2, NIR=7, CHM=9')
# # @click.option('--seeds',      type=click.Path(exists=True), required=True)
# # @click.option('--out-dir',    type=click.Path(),            required=True)
# # @click.option('--model',      default='vit_b',
# #               type=click.Choice(['vit_b', 'vit_l', 'vit_h']))
# # @click.option('--checkpoint', required=True,
# #               help='Path or URL to SAM .pth checkpoint')
# # @click.option('--tile-size',  default=512, show_default=True)
# # @click.option('--pad',        default=128, show_default=True)
# #
# # def main(image, seeds, out_dir, model, checkpoint, tile_size, pad):
# #     out_dir = Path(out_dir)
# #     if out_dir.exists():
# #         shutil.rmtree(out_dir)
# #     out_dir.mkdir(parents=True, exist_ok=True)
# #
# #     seeds_gdf = gpd.read_file(seeds)
# #
# #     with rasterio.open(image) as src:
# #         seeds_in = seeds_gdf[seeds_gdf.within(box(*src.bounds))].copy()
# #         if seeds_in.empty:
# #             raise SystemExit('No seeds overlap image extent.')
# #         click.echo(f'• seeds fed to SAM: {len(seeds_in):,}')
# #
# #         # cache (row,col) and world coords
# #         seeds_rc = [(src.index(pt.x, pt.y)[::-1] + pt.coords[0]) for pt in seeds_in.geometry]
# #
# #         core_sz = max(1, min(tile_size, 1024 - 2*pad))
# #         n_rows  = math.ceil(src.height / core_sz)
# #         n_cols  = math.ceil(src.width  / core_sz)
# #
# #         sam = SamGeo(model_type=model,
# #                      checkpoint=checkpoint,
# #                      n_extra_channels=1,   # NIR
# #                      automatic=False)
# #
# #         tile_ct = 0
# #         for ty in range(n_rows):
# #             for tx in range(n_cols):
# #                 row0, col0 = ty*core_sz, tx*core_sz
# #                 core_h = min(core_sz, src.height-row0)
# #                 core_w = min(core_sz, src.width -col0)
# #
# #                 # padded window
# #                 row_off = max(row0-pad, 0); col_off = max(col0-pad, 0)
# #                 row_end = min(row0+core_h+pad, src.height)
# #                 col_end = min(col0+core_w+pad, src.width)
# #                 h_p, w_p = row_end-row_off, col_end-col_off
# #                 window   = Window(col_off, row_off, w_p, h_p)
# #
# #                 # seed filter: keep those in core
# #                 core_bounds = box(*src.xy(row0, col0, offset='ul'),
# #                                    *src.xy(row0+core_h, col0+core_w, offset='lr'))
# #                 coords_tile = [[c-col_off, r-row_off]
# #                                for (r,c,_,_), pt in zip(seeds_rc, seeds_in.geometry)
# #                                if pt.within(core_bounds)]
# #                 if not coords_tile:
# #                     continue
# #
# #                 # read bands ------------------------------------------------
# #                 rgb16 = src.read([7, 5, 3], window=window).astype('float32')  # (3,h,w)
# #                 nir16 = src.read(8,        window=window).astype('float32')   # (h,w)
# #                 chm16 = src.read(9,        window=window).astype('float32')   # (h,w)
# #
# #                 # ground mask
# #                 ground = chm16 < CHM_GROUND
# #
# #                 # stretch to uint8 -----------------------------------------
# #                 rgb8 = np.empty_like(rgb16, dtype=np.uint8)
# #                 for i in range(3):
# #                     lo, hi = np.percentile(rgb16[i], RGB_PCTL)
# #                     rgb8[i] = np.clip((rgb16[i]-lo)*255/(hi-lo+1e-6), 0, 255)
# #                 nir8 = np.clip((nir16 - np.percentile(nir16, RGB_PCTL[0])) * 255 /
# #                                 (np.percentile(nir16, RGB_PCTL[1]) - np.percentile(nir16, RGB_PCTL[0]) + 1e-6),
# #                                 0, 255).astype(np.uint8)
# #
# #                 # apply ground mask
# #                 for i in range(3):
# #                     rgb8[i][ground] = 0
# #                 nir8[ground] = 0
# #
# #                 rgb_hwc = np.moveaxis(rgb8, 0, -1)  # (h,w,3)
# #                 nir_hwc = nir8[..., None]           # (h,w,1)
# #
# #                 # SAM inference --------------------------------------------
# #                 sam.set_image(rgb_hwc)
# #                 logits = sam.predict(
# #                             extra_channels=nir_hwc,
# #                             point_coords=coords_tile,
# #                             point_labels=[1]*len(coords_tile),
# #                             return_logits=True,
# #                             multimask_output=False,
# #                             return_results=True
# #                          )[2]                         # (N,h_p,w_p)
# #
# #                 # write logits ------------------------------------------------
# #                 # -------------------------------------------------------------
# #                 # add a background band: 1 where CHM < 3 m (ground), else 0
# #                 # guarantees ≥2 labels for graph-cut
# #                 # -------------------------------------------------------------
# #                 bg_band = ground.astype(np.float32)
# #
# #                 # make sure it matches logits’ H×W exactly (edge tiles)
# #                 h_out, w_out = logits.shape[1:]
# #                 bg_band = bg_band[:h_out, :w_out]  # crop if larger
# #
# #                 if bg_band.shape != (h_out, w_out):  # pad if smaller
# #                     pad = np.zeros((h_out, w_out), np.float32)
# #                     pad[:bg_band.shape[0], :bg_band.shape[1]] = bg_band
# #                     bg_band = pad
# #
# #                 logits = np.concatenate([logits, bg_band[None, ...]], axis=0)
# #
# #
# #
# #                 log_path = out_dir / f'sam_logits_r{row0}_c{col0}.tif'
# #                 prof = src.profile
# #
# #                 prof.update(driver='GTiff', compress='deflate', dtype='float32', nodata=None,
# #                             count = logits.shape[0],
# #                             width = w_p, height = h_p,
# #                             transform = (Affine.translation(src.transform.c + col_off * src.transform.a,
# #                                                 src.transform.f + row_off * src.transform.e)
# #                              * Affine.scale(src.transform.a, src.transform.e))
# #                             )
# #                 with rasterio.open(log_path, 'w', **prof) as dst:
# #                     dst.write(logits.astype('float32'))
# #                 click.echo(f'  • wrote {log_path.name}')
# #                 tile_ct += 1
# #
# #         click.echo(f'✓ {tile_ct} logit tiles written → {out_dir}')
# #
# # if __name__ == '__main__':
# #     try:
# #         main()
# #     except Exception as exc:
# #         click.echo(f'Error: {exc}', err=True)
# #         sys.exit(1)
