#!/usr/bin/env python3
"""
make_canonical_seeds.py  (REV‑14b • cost‑aware clustering with XY‑threshold)

Writes a layer “canonical_seeds” where one DBSCAN *cluster* ≈ one crown.

Distance used in stage‑2 (between seeds *i* and *j*):

    d_eff = xy_dist · (1 + W · 𝕴{xy_dist > xy_thresh} · mean_cost)

Given:   xy_dist   [m]          – Euclidean ground distance
         mean_cost [0–1]        – average cost along straight line
         W         (--cost-weight)
         xy_thresh (--xy-thresh) – distance below which cost is ignored

New flags
─────────
--xy-thresh <m>   (default 0)  ·  ignore the cost term for extremely
                                 close seeds so they always merge.

--debug-dist      ·  prints min/median/max d_eff before DBSCAN so you
                     can judge if --merge-radius is sensible.

Everything else (stage‑1 thinning, dz‑split, multiplicity cap, NMS)
works exactly like REV‑14.
"""
from __future__ import annotations
from pathlib import Path
import sys, math, click
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from rasterio.transform import rowcol

KEEP = ["geometry", "height", "origin"]           # columns to keep
NODATA_COST = 1.0                                 # treat nodata as high cost


# ───────────────────────── small helpers ───────────────────────────
def add_chm_height(gdf: gpd.GeoDataFrame, chm_path: str | Path) -> gpd.GeoDataFrame:
    """Sample CHM for points lacking a 'height' field."""
    with rasterio.open(chm_path) as src:
        vals = np.array(
            [v[0] if v[0] is not np.ma.masked else np.nan
             for v in src.sample([(p.x, p.y) for p in gdf.geometry])],
            np.float32)
    gdf["height"] = vals
    return gdf.dropna(subset=["height"])


def nms_per_crown(gdf: gpd.GeoDataFrame, base_r: float, scale_r: float
                  ) -> gpd.GeoDataFrame:
    """Adaptive NMS inside each crown cluster (keep tallest seed)."""
    if base_r <= 0 and scale_r <= 0:
        return gdf

    kept: list[gpd.GeoDataFrame] = []
    for _, sub in gdf.groupby("cluster"):
        sub = sub.sort_values("height", ascending=False).copy()
        pts = np.c_[sub.geometry.x, sub.geometry.y]
        tree = cKDTree(pts)
        keep = np.zeros(len(sub), bool)

        for i, (x, y, h) in enumerate(zip(pts[:, 0], pts[:, 1], sub.height)):
            if keep[i]:
                continue
            keep[i] = True
            r = max(base_r, scale_r * h)
            keep[tree.query_ball_point([x, y], r)] = False
            keep[i] = True                    # re‑enable winner
        kept.append(sub[keep])

    return pd.concat(kept, ignore_index=True)


# ───────────────── cost‑aware distance matrix ──────────────────────
def build_distance_matrix(xs: np.ndarray, ys: np.ndarray,
                          cost: np.ndarray, tfm, weight: float,
                          xy_thresh: float, samples: int = 8) -> np.ndarray:
    """
    Pre‑compute a symmetric matrix D where
        D[i,j] = xy_dist * (1 + weight * mean_cost)
    unless xy_dist <= xy_thresh  →  cost term is ignored.
    """
    n = len(xs)
    D = np.zeros((n, n), np.float32)

    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n):
            dx, dy = xs[j] - xi, ys[j] - yi
            xy_dist = math.hypot(dx, dy)
            if xy_dist == 0:
                continue
            if xy_dist <= xy_thresh or weight == 0:
                D[i, j] = D[j, i] = xy_dist
                continue

            # sample points along the segment (exclude endpoints)
            ts = np.linspace(0.0, 1.0, samples + 2, dtype=np.float32)[1:-1]
            xs_line = xi + ts * dx
            ys_line = yi + ts * dy
            rows, cols = rowcol(tfm, xs_line, ys_line, op=float)
            rows = np.clip(rows.round().astype(int), 0, cost.shape[0] - 1)
            cols = np.clip(cols.round().astype(int), 0, cost.shape[1] - 1)
            mean_cost = cost[rows, cols].mean()
            D[i, j] = D[j, i] = xy_dist * (1.0 + weight * mean_cost)

    return D


# ───────────────────────────── CLI ─────────────────────────────────
@click.command(context_settings={"show_default": True})
# inputs
@click.option("--chm-seeds",   type=click.Path(exists=True), required=True)
@click.option("--den-seeds",   type=click.Path(exists=True), required=True)
@click.option("--chm-raster",  type=click.Path(exists=True), required=True)
@click.option("--cost-surface", type=click.Path(exists=True), required=True,
              help="Single‑band cost raster (0…1; low = easy to merge).")
@click.option("--out", "out_path", type=click.Path(), required=True)
# stage‑1 thinning
@click.option("--eps-scale",   default=0.4,  help="ε = k·height  (m/m)")
@click.option("--min-eps",     default=2.0)
@click.option("--max-eps",     default=8.0)
@click.option("--z-thresh",    default=-1.0,
              help="Reject cluster if Δh > z‑thresh (‑1 = off)")
@click.option("--min-samples", default=2)
# stage‑2 merge hyper‑parameters
@click.option("--merge-radius", default=1.5, help="ε in metres after cost")
@click.option("--cost-weight",  default=0.5, help="W in d_eff")
@click.option("--xy-thresh",    default=0.8,
              help="Ignore cost when xy distance ≤ this (m)")
@click.option("--dz-merge",     default=0.0,
              help="Split crown if Δh > dz‑merge after merge (0 = off)")
# multiplicity
@click.option("--keep-all-stage1", is_flag=True,
              help="Forward *all* stage‑1 seeds (skip tallest‑only).")
@click.option("--stage1-top",  default=1,
              help="If not all, N tallest per stage‑1 cluster.")
@click.option("--max-per-cluster", default=0,
              help="Cap seeds / crown (0 = unlimited).")
# NMS
@click.option("--nms-base",  default=1.0)
@click.option("--nms-scale", default=0.06)
# debug
@click.option("--debug-dist", is_flag=True,
              help="Print min/median/max effective distances.")
def main(chm_seeds, den_seeds, chm_raster, cost_surface, out_path,
         eps_scale, min_eps, max_eps, z_thresh, min_samples,
         merge_radius, cost_weight, xy_thresh, dz_merge,
         keep_all_stage1, stage1_top, max_per_cluster,
         nms_base, nms_scale, debug_dist):

    # ── 0. load detections ─────────────────────────────────────────
    chm = gpd.read_file(chm_seeds); chm["origin"] = "chm"
    den = gpd.read_file(den_seeds); den["origin"] = "density"
    chm.rename(columns={"ch_max": "height"}, inplace=True)
    den.rename(columns={"den_max": "height"}, inplace=True)
    if "height" not in chm: chm = add_chm_height(chm, chm_raster)
    if "height" not in den: den = add_chm_height(den, chm_raster)

    seeds = gpd.GeoDataFrame(pd.concat([chm[KEEP], den[KEEP]], ignore_index=True),
                             geometry="geometry", crs=chm.crs)
    if seeds.empty:
        click.echo("No seeds after CHM sampling.", err=True)
        sys.exit(1)

    seeds["x"], seeds["y"] = seeds.geometry.x, seeds.geometry.y
    pts_xy = seeds[["x", "y"]].values
    tree = cKDTree(pts_xy)

    # ── 1. stage‑1 adaptive‑ε clustering (optional thinning) ──────
    cl1 = -np.ones(len(seeds), int); cid = 0
    for i in range(len(seeds)):
        if cl1[i] != -1:
            continue
        eps = float(np.clip(eps_scale * seeds.height.iloc[i], min_eps, max_eps))
        idx = tree.query_ball_point(pts_xy[i], eps)
        if z_thresh >= 0 and seeds.height.iloc[idx].ptp() > z_thresh:
            continue
        if len(idx) >= min_samples:
            cl1[idx] = cid; cid += 1
    seeds["cluster1"] = cl1

    if keep_all_stage1:
        stage1 = seeds.copy()
    else:
        stage1_top = max(1, stage1_top)
        topfun = lambda df: df.nlargest(stage1_top, "height")
        tall = (seeds[seeds.cluster1 != -1]
                .groupby("cluster1", group_keys=False).apply(topfun))
        single = seeds[seeds.cluster1 == -1]
        stage1 = pd.concat([tall, single], ignore_index=True)

    # ── 2. load cost raster ────────────────────────────────────────
    with rasterio.open(cost_surface) as src:
        cost_arr = src.read(1).astype(np.float32)
        cost_arr[cost_arr == src.nodatavals[0]] = NODATA_COST
        cost_tfm = src.transform

    xs, ys = stage1["x"].values, stage1["y"].values
    D = build_distance_matrix(xs, ys, cost_arr, cost_tfm,
                              cost_weight, xy_thresh, samples=12)

    if debug_dist:
        dvals = D[np.triu_indices(len(D), 1)]
        click.echo(f"d_eff  min/median/max = "
                   f"{dvals.min():.2f} / {np.median(dvals):.2f} / {dvals.max():.2f}")

    # ── 3. DBSCAN on pre‑computed cost‑aware distances ─────────────
    db = DBSCAN(eps=merge_radius, min_samples=1,
                metric="precomputed").fit(D)
    stage1["cluster"] = db.labels_

    # optional Δh split
    if dz_merge > 0:
        parts, new_id = [], 0
        for cid, sub in stage1.groupby("cluster"):
            if sub.height.ptp() <= dz_merge:
                sub["cluster"] = new_id; parts.append(sub); new_id += 1
            else:
                mid = sub.height.median()
                for g in (sub[sub.height <= mid], sub[sub.height > mid]):
                    if not g.empty:
                        g["cluster"] = new_id; parts.append(g); new_id += 1
        stage1 = pd.concat(parts, ignore_index=True)

    # ── 4. cap multiplicity per crown ──────────────────────────────
    def _trim(df):
        if max_per_cluster <= 0 or len(df) <= max_per_cluster:
            return df
        return df.nlargest(max_per_cluster, "height")

    trimmed = (stage1.groupby("cluster", group_keys=False)
               .apply(_trim).reset_index(drop=True))

    # ── 5. adaptive NMS inside crown ───────────────────────────────
    final = nms_per_crown(trimmed, nms_base, nms_scale)

    # ── 6. write GeoPackage layer ──────────────────────────────────
    final = final.rename(columns={"height": "ch_max"})
    final.insert(0, "id", range(len(final)))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

    final[["id", "cluster", "ch_max", "origin", "geometry"]].to_file(
        out_path, layer="canonical_seeds", driver="GPKG", overwrite=True)

    click.echo(f"✓ canonical seeds: {len(final):,}  →  {out_path}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

# #!/usr/bin/env python3
# """
# make_canonical_seeds.py  (REV‑9 • stricter + 2‑stage DBSCAN)
# ─────────────────────────────────────────────────────────────
# Output  ➜  GeoPackage layer  “canonical_seeds”
#
# Processing pipeline
# -------------------
#   1. height‑adaptive ε‑clustering        (scipy cKDTree)
#   2. fixed‑radius DBSCAN  (ε = --merge-radius) on stage‑1 centroids
#   3. adaptive spatial NMS  (optional)           – keeps tallest seed
# """
# # --------------------------------------------------------------------
# from pathlib import Path
# import sys, warnings, click
# import geopandas as gpd, pandas as pd, numpy as np
# import rasterio
# from scipy.spatial import cKDTree
# from sklearn.cluster import DBSCAN
#
# KEEP_COLS = ["geometry", "height", "origin"]      # kept for output
#
# # ───────────────────────── helpers ─────────────────────────────────
# def add_chm_height(gdf: gpd.GeoDataFrame, chm_path: Path) -> gpd.GeoDataFrame:
#     """Sample CHM at point locations → add/overwrite `height` column (m)."""
#     with rasterio.open(chm_path) as src:
#         vals = np.array([v[0] if v[0] is not np.ma.masked else np.nan
#                          for v in src.sample([(p.x, p.y) for p in gdf.geometry])],
#                         dtype=np.float32)
#     gdf["height"] = vals
#     return gdf.dropna(subset=["height"])
#
#
# def adaptive_nms(gdf: gpd.GeoDataFrame, base_r: float, scale: float) -> gpd.GeoDataFrame:
#     """Non‑max suppression; radius = max(base_r , scale·height)."""
#     if base_r <= 0 and scale <= 0:
#         return gdf                               # NMS disabled
#     gdf = gdf.sort_values("height", ascending=False).copy()
#     coords  = np.c_[gdf.geometry.x, gdf.geometry.y]
#     heights = gdf.height.values
#     tree    = cKDTree(coords)
#
#     keep = np.zeros(len(gdf), bool)
#     for idx, (x, y, h) in enumerate(zip(coords[:, 0], coords[:, 1], heights)):
#         if keep[idx]:
#             continue
#         keep[idx] = True                         # keep the current tallest
#         r = max(base_r, scale * h)
#         neigh = tree.query_ball_point([x, y], r=r)
#         keep[neigh] = False
#         keep[idx]   = True                       # restore current
#     return gdf.loc[keep].reset_index(drop=True)
#
# # -------------------------------------------------------------------
# @click.command()
# #  input layers -----------------------------------------------------
# @click.option("--chm-seeds",  type=click.Path(exists=True), required=True,
#               help="Point layer from CHM peaks (e.g. LocalMax).")
# @click.option("--den-seeds",  type=click.Path(exists=True), required=True,
#               help="Point layer from density‑based detector.")
# @click.option("--chm-raster", type=click.Path(exists=True), required=True,
#               help="Canopy‑height model (for missing heights).")
# @click.option("--out", "out_path", type=click.Path(), required=True,
#               help="Output GeoPackage path.")
# #  stage‑1 adaptive ε‑clustering ------------------------------------
# @click.option("--eps-scale", default=0.25, show_default=True,
#               help="k in ε = k·height  (m/m).")
# @click.option("--min-eps",   default=2.0,  show_default=True,
#               help="Minimum ε (m).")
# @click.option("--max-eps",   default=6.0,  show_default=True,
#               help="Maximum ε (m).")
# @click.option("--z-thresh",  default=1.0,  show_default=True,
#               help="Max height range allowed inside a cluster (m).")
# @click.option("--min-samples", default=3,   show_default=True,
#               help="Min raw seeds to form a stage‑1 cluster.")
# #  stage‑2 fixed‑radius DBSCAN --------------------------------------
# @click.option("--merge-radius", default=1.5, show_default=True,
#               help="Radius (m) for the second DBSCAN merge.")
# #  NMS --------------------------------------------------------------
# @click.option("--nms-base",  default=1.0,  show_default=True,
#               help="Base NMS radius for small crowns (m).")
# @click.option("--nms-scale", default=0.06, show_default=True,
#               help="Extra radius per metre of height (m/m).")
# def main(chm_seeds, den_seeds, chm_raster, out_path,
#          eps_scale, min_eps, max_eps,
#          z_thresh, min_samples,
#          merge_radius,
#          nms_base, nms_scale):
#
#     # ---------- read & harmonise -----------------------------------
#     chm_gdf = gpd.read_file(chm_seeds); chm_gdf["origin"] = "chm"
#     den_gdf = gpd.read_file(den_seeds); den_gdf["origin"] = "density"
#
#     chm_gdf.rename(columns={"ch_max": "height"}, inplace=True)
#     den_gdf.rename(columns={"den_max": "height"}, inplace=True)
#
#     if "height" not in chm_gdf.columns:
#         chm_gdf = add_chm_height(chm_gdf, chm_raster)
#     if "height" not in den_gdf.columns:
#         den_gdf = add_chm_height(den_gdf, chm_raster)
#
#     seeds = pd.concat([chm_gdf[KEEP_COLS], den_gdf[KEEP_COLS]],
#                       ignore_index=True)
#     seeds = gpd.GeoDataFrame(seeds, geometry="geometry", crs=chm_gdf.crs)
#
#     if seeds.empty:
#         raise SystemExit("All seed points lost after NaN‑height removal.")
#
#     # ---------- stage‑1 adaptive ε‑balls ---------------------------
#     seeds["x"], seeds["y"] = seeds.geometry.x, seeds.geometry.y
#     coords = seeds[["x", "y"]].to_numpy()
#     tree   = cKDTree(coords)
#
#     cluster = -np.ones(len(seeds), dtype=int)
#     cid     = 0
#     for i in range(len(seeds)):
#         if cluster[i] != -1:
#             continue
#         eps_i = np.clip(eps_scale * seeds.height.iloc[i], min_eps, max_eps)
#         idx   = tree.query_ball_point(coords[i], eps_i)
#         # height consistency
#         if z_thresh >= 0:
#             if (seeds.height.iloc[idx].max() - seeds.height.iloc[idx].min()) > z_thresh:
#                 continue
#         if len(idx) >= min_samples:
#             cluster[idx] = cid
#             cid += 1
#     seeds["cluster1"] = cluster
#
#     # ---------- pick tallest point per stage‑1 cluster -------------
#     sel_idx = (
#         seeds[seeds.cluster1 != -1].groupby("cluster1").height.idxmax()
#         .tolist() +
#         seeds[seeds.cluster1 == -1].index.tolist()
#     )
#     stage1 = seeds.loc[sel_idx].reset_index(drop=True)
#
#     # ---------- stage‑2 fixed DBSCAN merge  ------------------------
#     db = DBSCAN(eps=merge_radius, min_samples=1).fit(stage1[["x", "y"]])
#     stage1["cluster2"] = db.labels_
#     stage2 = stage1.loc[stage1.groupby("cluster2").height.idxmax()]
#     stage2 = stage2.reset_index(drop=True)
#
#     # ---------- adaptive NMS  --------------------------------------
#     canon = adaptive_nms(stage2, base_r=nms_base, scale=nms_scale)
#
#     # ---------- final tidy‑up & write ------------------------------
#     canon = canon.rename(columns={"height": "ch_max"})
#     canon.insert(0, "id", range(len(canon)))
#
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     canon.to_file(out_path, layer="canonical_seeds",
#                   driver="GPKG", overwrite=True)
#
#     click.echo(f"✓ canonical seeds: {len(canon):,}  →  {out_path}")
#
#
# # -------------------------------------------------------------------
# if __name__ == "__main__":
#     try:
#         main()                        # pylint: disable=no-value-for-parameter
#     except Exception as exc:
#         click.echo(f"Error: {exc}", err=True)
#         sys.exit(1)
#
# # #!/usr/bin/env python3
# # """
# # make_canonical_seeds.py  (REV-8 – adaptive NMS)
# # ────────────────────────────────────────────────────────────
# # Creates *one (or a few) seed(s) per crown* in two stages:
# #
# # 1. **Height-adaptive agglomeration** ─ merge seeds within an
# #    ε-ball where ε = k · height, clipped to [`min_eps`,`max_eps`].
# # 2. **Adaptive Spatial NMS** ─ after merging, keep only the tallest
# #    seed inside a radius that grows with crown height:
# #       radius  =  max(base_radius , scale · height)
# #
# # CLI flags
# # ---------
# # --eps-scale   k        (default 0.15)
# # --min-eps     metres   (default 1.0)
# # --max-eps     metres   (default 3.5)
# # --nms-base    metres   fixed radius for small crowns  (default 1.2)
# # --nms-scale   m / m    extra radius per metre height   (default 0.07)
# #
# # Set `--nms-base 0 --nms-scale 0` to disable the final NMS.
# # """
# # from pathlib import Path
# # import sys
# #
# # import click
# # import geopandas as gpd
# # import numpy as np
# # import pandas as pd
# # import rasterio
# # from scipy.spatial import cKDTree
# #
# # KEEP_COLS = ["geometry", "height", "origin"]
# #
# # # ───────────────── helper funcs ─────────────────────────
# #
# # def add_chm_height(gdf: gpd.GeoDataFrame, chm_path: Path) -> gpd.GeoDataFrame:
# #     """Sample CHM at point locations and add `height` column."""
# #     with rasterio.open(chm_path) as src:
# #         vals = [v[0] if v[0] is not np.ma.masked else np.nan
# #                 for v in src.sample([(p.x, p.y) for p in gdf.geometry])]
# #     gdf["height"] = vals
# #     return gdf.dropna(subset=["height"])
# #
# #
# # def adaptive_nms(gdf: gpd.GeoDataFrame, base_r: float, scale: float) -> gpd.GeoDataFrame:
# #     """Non-max suppression with radius = max(base_r, scale*height)."""
# #     if base_r <= 0 and scale <= 0:
# #         return gdf  # NMS disabled
# #     gdf = gdf.sort_values("ch_max", ascending=False).copy()
# #     coords = np.c_[gdf.geometry.x, gdf.geometry.y]
# #     heights = gdf.ch_max.values
# #     tree = cKDTree(coords)
# #     keep = np.zeros(len(gdf), bool)
# #     for i, (x, y, h) in enumerate(zip(coords[:, 0], coords[:, 1], heights)):
# #         if keep[i]:
# #             continue
# #         keep[i] = True
# #         rad = max(base_r, scale * h)
# #         neigh = tree.query_ball_point([x, y], r=rad)
# #         keep[neigh] = False
# #         keep[i] = True  # keep the focal (tallest) seed
# #     return gdf[keep].reset_index(drop=True)
# #
# # # ───────────────────── CLI ──────────────────────────────
# #
# # @click.command()
# # @click.option("--chm-seeds",  type=click.Path(exists=True), required=True)
# # @click.option("--den-seeds",  type=click.Path(exists=True), required=True)
# # @click.option("--chm-raster", type=click.Path(exists=True), required=True)
# # @click.option("--out",        "out_path", type=click.Path(), required=True)
# #
# # @click.option("--eps-scale",  default=0.15, show_default=True,
# #               help="k in ε = k·height (m/m).")
# # @click.option("--min-eps",    default=1.0,  show_default=True, help="Minimum ε (m).")
# # @click.option("--max-eps",    default=3.5,  show_default=True, help="Maximum ε (m).")
# #
# # @click.option("--z-thresh",   default=-1.0, show_default=True,
# #               help="Max height range inside cluster; -1 disables.")
# # @click.option("--min-samples", default=1,    show_default=True,
# #               help="Min seeds inside ε-ball to form a cluster.")
# #
# # @click.option("--nms-base",   default=1.2,  show_default=True,
# #               help="Base NMS radius for small crowns (m).")
# # @click.option("--nms-scale",  default=0.07, show_default=True,
# #               help="Additional radius per metre of height (m/m).")
# #
# # def main(chm_seeds, den_seeds, chm_raster, out_path,
# #          eps_scale, min_eps, max_eps,
# #          z_thresh, min_samples,
# #          nms_base, nms_scale):
# #     # 1 ─ Load --------------------------------------------------------
# #     chm_gdf = gpd.read_file(chm_seeds); chm_gdf["origin"] = "chm"
# #     den_gdf = gpd.read_file(den_seeds); den_gdf["origin"] = "density"
# #
# #     chm_gdf.rename(columns={"ch_max": "height"}, inplace=True)
# #     den_gdf.rename(columns={"den_max": "height"}, inplace=True)
# #
# #     if "height" not in chm_gdf.columns:
# #         chm_gdf = add_chm_height(chm_gdf, chm_raster)
# #     if "height" not in den_gdf.columns:
# #         den_gdf = add_chm_height(den_gdf, chm_raster)
# #
# #     seeds = pd.concat([chm_gdf[KEEP_COLS], den_gdf[KEEP_COLS]], ignore_index=True)
# #     seeds = gpd.GeoDataFrame(seeds, geometry="geometry", crs=chm_gdf.crs)
# #
# #     # 2 ─ Adaptive-ε clustering --------------------------------------
# #     seeds["x"], seeds["y"] = seeds.geometry.x, seeds.geometry.y
# #     coords = seeds[["x", "y"]].to_numpy()
# #     tree = cKDTree(coords)
# #
# #     n = len(seeds)
# #     visited = np.zeros(n, bool)
# #     cluster = -np.ones(n, int)
# #     cid = 0
# #
# #     for i in range(n):
# #         if visited[i]:
# #             continue
# #         eps_i = np.clip(eps_scale * seeds.height.iloc[i], min_eps, max_eps)
# #         idx = tree.query_ball_point(coords[i], eps_i)
# #         height_ok = (z_thresh < 0) or (
# #             seeds.height.iloc[idx].max() - seeds.height.iloc[idx].min() <= z_thresh)
# #         if len(idx) >= min_samples and height_ok:
# #             cluster[idx] = cid; visited[idx] = True; cid += 1
# #         else:
# #             visited[i] = True
# #     seeds["cluster"] = cluster
# #
# #     # 3 ─ Tallest per cluster + singletons ---------------------------
# #     tall_idx = (
# #         seeds[seeds.cluster != -1].groupby("cluster").height.idxmax().tolist() +
# #         seeds[seeds.cluster == -1].index.tolist())
# #     canon = seeds.loc[tall_idx].copy()
# #     canon.insert(0, "id", range(len(canon)))
# #     canon.rename(columns={"height": "ch_max"}, inplace=True)
# #
# #     # 4 ─ Adaptive NMS ----------------------------------------------
# #     canon = adaptive_nms(canon, base_r=nms_base, scale=nms_scale)
# #
# #     # 5 ─ Write ------------------------------------------------------
# #     out_path = Path(out_path)
# #     out_path.parent.mkdir(parents=True, exist_ok=True)
# #     canon.to_file(out_path, layer="canonical_seeds", driver="GPKG", overwrite=True)
# #     click.echo(f"✓ canonical seeds: {len(canon):,} → {out_path}")
# #
# #
# # if __name__ == "__main__":
# #     try:
# #         main()  # pylint: disable=no-value-for-parameter
# #     except Exception as e:
# #         click.echo(f"Error: {e}", err=True)
# #         sys.exit(1)
