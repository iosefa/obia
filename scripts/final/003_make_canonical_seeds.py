#!/usr/bin/env python3
"""
make_canonical_seeds.py  (REV-14c • sparse cost-aware clustering)
-----------------------------------------------------------------
Implements Approach A: build a *sparse* neighbour graph instead of
an n² dense matrix so tens–hundreds of thousands of seeds fit easily
in RAM.

Distance (same as before):

    d_eff = xy_dist · (1 + W · 𝟙{xy_dist > xy_thresh} · mean_cost)

If d_eff ≤ --merge-radius, an undirected edge is stored; DBSCAN runs
on the resulting sparse, pre-computed distance matrix.
"""
from __future__ import annotations
from pathlib import Path
import sys, math, click
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
from rasterio.transform import rowcol

KEEP = ["geometry", "height", "origin"]
NODATA_COST = 1.0            # treat nodata as high cost (hard to merge)

# ───────────────────────── helpers (unchanged) ────────────────────
def add_chm_height(gdf: gpd.GeoDataFrame, chm_path: str | Path) -> gpd.GeoDataFrame:
    with rasterio.open(chm_path) as src:
        vals = np.array(
            [v[0] if v[0] is not np.ma.masked else np.nan
             for v in src.sample([(p.x, p.y) for p in gdf.geometry])],
            np.float32)
    gdf["height"] = vals
    return gdf.dropna(subset=["height"])

def nms_per_crown(gdf: gpd.GeoDataFrame, base_r: float, scale_r: float) -> gpd.GeoDataFrame:
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
            keep[i] = True
        kept.append(sub[keep])
    return pd.concat(kept, ignore_index=True)

# ───────────── sparse cost-aware neighbour graph builder ───────────
def build_sparse_distance_graph(xs: np.ndarray, ys: np.ndarray,
                                cost: np.ndarray, tfm,
                                weight: float, xy_thresh: float,
                                merge_radius: float,
                                samples: int = 8) -> "coo_matrix":
    """
    Returns a symmetric COO sparse matrix holding d_eff for *pairs*
    whose cost-aware distance ≤ merge_radius.
    Complexity:  O(n · k)   where k ≈ average neighbours within search radius.
    Memory   :  O(n · k)
    """
    n = len(xs)
    rows, cols, data = [], [], []
    kd = cKDTree(np.c_[xs, ys])
    search_r = merge_radius + xy_thresh + 1e-6  # include pairs that *might* pass

    # pre-compute sampling fractions along a segment
    ts = np.linspace(0.0, 1.0, samples + 2, dtype=np.float32)[1:-1]

    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in kd.query_ball_point([xi, yi], search_r):
            if j <= i:                     # upper-triangle only
                continue
            dx, dy = xs[j] - xi, ys[j] - yi
            xy_dist = math.hypot(dx, dy)
            if xy_dist == 0.0:
                continue

            if xy_dist <= xy_thresh or weight == 0.0:
                d_eff = xy_dist
            else:
                # sample cost along straight line
                xs_line = xi + ts * dx
                ys_line = yi + ts * dy
                rows_pix, cols_pix = rowcol(tfm, xs_line, ys_line, op=float)
                rows_pix = np.clip(rows_pix.round().astype(int), 0, cost.shape[0]-1)
                cols_pix = np.clip(cols_pix.round().astype(int), 0, cost.shape[1]-1)
                mean_cost = cost[rows_pix, cols_pix].mean()
                d_eff = xy_dist * (1.0 + weight * mean_cost)

            if d_eff <= merge_radius:
                rows.extend((i, j))
                cols.extend((j, i))
                data.extend((d_eff, d_eff))

    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

# ───────────────────────────── CLI ─────────────────────────────────
@click.command(context_settings={"show_default": True})
# inputs
@click.option("--chm-seeds",   type=click.Path(exists=True), required=True)
@click.option("--den-seeds",   type=click.Path(exists=True), required=True)
@click.option("--chm-raster",  type=click.Path(exists=True), required=True)
@click.option("--cost-surface", type=click.Path(exists=True), required=True,
              help="Single-band cost raster (0…1; low = easy to merge).")
@click.option("--out", "out_path", type=click.Path(), required=True)
# stage-1 thinning
@click.option("--eps-scale",   default=0.4,  help="ε = k·height  (m/m)")
@click.option("--min-eps",     default=2.0)
@click.option("--max-eps",     default=8.0)
@click.option("--z-thresh",    default=-1.0,
              help="Reject cluster if Δh > z-thresh (-1 = off)")
@click.option("--min-samples", default=2)
# stage-2 merge hyper-parameters
@click.option("--merge-radius", default=1.5, help="ε in metres after cost")
@click.option("--cost-weight",  default=0.5, help="W in d_eff")
@click.option("--xy-thresh",    default=0.8,
              help="Ignore cost when xy distance ≤ this (m)")
@click.option("--samples",      default=8, show_default=True,
              help="# cost samples per edge (lower = faster)")
@click.option("--dz-merge",     default=0.0,
              help="Split crown if Δh > dz-merge after merge (0 = off)")
# multiplicity
@click.option("--keep-all-stage1", is_flag=True,
              help="Forward *all* stage-1 seeds (skip tallest-only).")
@click.option("--stage1-top",  default=1,
              help="If not all, N tallest per stage-1 cluster.")
@click.option("--max-per-cluster", default=0,
              help="Cap seeds / crown (0 = unlimited).")
# NMS
@click.option("--nms-base",  default=1.0)
@click.option("--nms-scale", default=0.06)
# debug
@click.option("--debug-dist", is_flag=True,
              help="Print min/median/max d_eff before DBSCAN.")
def main(chm_seeds, den_seeds, chm_raster, cost_surface, out_path,
         eps_scale, min_eps, max_eps, z_thresh, min_samples,
         merge_radius, cost_weight, xy_thresh, samples, dz_merge,
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

    # ── 1. stage-1 adaptive-ε clustering (optional thinning) ──────
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
        tall = (seeds[seeds.cluster1 != -1]
                .groupby("cluster1", group_keys=False)
                .apply(lambda df: df.nlargest(stage1_top, "height")))
        single = seeds[seeds.cluster1 == -1]
        stage1 = pd.concat([tall, single], ignore_index=True)

    # ── 2. load cost raster ────────────────────────────────────────
    with rasterio.open(cost_surface) as src:
        cost_arr = src.read(1).astype(np.float32)
        cost_arr[cost_arr == src.nodatavals[0]] = NODATA_COST
        cost_tfm = src.transform

    xs, ys = stage1["x"].values, stage1["y"].values
    G = build_sparse_distance_graph(xs, ys, cost_arr, cost_tfm,
                                    cost_weight, xy_thresh,
                                    merge_radius, samples=samples)

    if debug_dist and G.nnz:
        dvals = G.data
        click.echo(f"d_eff  min/median/max = "
                   f"{dvals.min():.2f} / {np.median(dvals):.2f} / {dvals.max():.2f}")

    # ── 3. DBSCAN on sparse pre-computed distances ────────────────
    db = DBSCAN(eps=merge_radius, min_samples=1,
                metric="precomputed", n_jobs=-1).fit(G)
    stage1["cluster"] = db.labels_

    # optional Δh split
    if dz_merge > 0:
        parts, new_id = [], 0
        for _, sub in stage1.groupby("cluster"):
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

    # ── 6. write output ────────────────────────────────────────────
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