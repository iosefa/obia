from __future__ import annotations
from pathlib import Path
import sys, math, click
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.transform import rowcol, xy
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.cluster import DBSCAN


def _detect_chm_peaks(arr: np.ndarray,
                 h_min: float,
                 min_dist_px: int,
                 sigma: float = 0) -> np.ndarray:
    """Return (row, col) indices of local maxima in *arr*."""
    if sigma > 0:
        arr = gaussian_filter(arr, sigma=sigma)

    local_max = (arr == maximum_filter(arr, size=2 * min_dist_px + 1))
    peaks = np.logical_and(local_max, arr >= h_min)

    return np.column_stack(np.where(peaks))


def _detect_den_peaks(arr: np.ndarray,
                 v_min: float,
                 min_dist_px: int,
                 sigma: int = 0) -> np.ndarray:
    """Return (row, col) indices of local maxima in *arr* ≥ v_min."""
    if sigma > 0:
        arr = gaussian_filter(arr, sigma=sigma)

    local_max = (arr == maximum_filter(arr, size=2 * min_dist_px + 1))
    peaks = np.logical_and(local_max, arr >= v_min)
    return np.column_stack(np.where(peaks))


def make_density_seeds(density_raster, seeds_gpkg, d_min=4.5,
                       min_dist_px=4, gauss_sigma=2) -> None:
    raster_path = Path(density_raster)
    if not raster_path.exists():
        raise SystemExit(f"✗ density raster not found: {raster_path}")

    print("• reading density raster …")
    with rasterio.open(raster_path) as src:
        den = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
        crs = src.crs

    print("• detecting peaks …")
    peak_rc = _detect_den_peaks(den, d_min, min_dist_px, gauss_sigma)
    if peak_rc.size == 0:
        raise SystemExit("No density peaks found — lower D_MIN or check raster.")

    rows, cols = peak_rc[:, 0], peak_rc[:, 1]
    xs, ys = xy(transform, rows, cols, offset="center")
    dvals = den[rows, cols]

    gdf = gpd.GeoDataFrame(
        {"id": np.arange(len(xs)), "den_max": dvals},
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs,
    )

    out_path = Path(seeds_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", overwrite=True)

    print(f"✓ wrote {len(gdf):,} density-seed points → {out_path}")


def make_chm_seeds(chm_raster, seeds_gpkg, h_min_m=2.5, min_dist_px=3, gauss_sigma=1) -> None:
    chm_path = Path(chm_raster)
    if not chm_path.exists():
        raise SystemExit(f"✗ CHM raster not found: {chm_path}")

    print("• reading CHM raster …")
    with rasterio.open(chm_path) as src:
        chm = src.read(1, masked=True).filled(np.nan)
        transform = src.transform
        crs = src.crs

    print("• detecting peaks …")
    peak_rc = _detect_chm_peaks(chm, h_min_m, min_dist_px, gauss_sigma)
    if peak_rc.size == 0:
        raise SystemExit("No peaks found – adjust H_MIN_M or check CHM.")

    rows, cols = peak_rc[:, 0], peak_rc[:, 1]
    xs, ys = xy(transform, rows, cols, offset="center")
    heights = chm[rows, cols]

    gdf = gpd.GeoDataFrame(
        {"id": np.arange(len(xs)), "ch_max": heights},
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs,
    )

    out_path = Path(seeds_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", overwrite=True)

    print(f"✓ wrote {len(gdf):,} CHM seed points → {out_path}")


def _add_chm_height(gdf: gpd.GeoDataFrame, chm_path: str | Path) -> gpd.GeoDataFrame:
    with rasterio.open(chm_path) as src:
        vals = np.array(
            [v[0] if v[0] is not np.ma.masked else np.nan
             for v in src.sample([(p.x, p.y) for p in gdf.geometry])],
            np.float32)
    gdf["height"] = vals
    return gdf.dropna(subset=["height"])


def _nms_per_crown(gdf: gpd.GeoDataFrame, base_r: float, scale_r: float
                  ) -> gpd.GeoDataFrame:
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


def _build_distance_matrix(xs: np.ndarray, ys: np.ndarray,
                          cost: np.ndarray, tfm, weight: float,
                          xy_thresh: float, samples: int = 8) -> np.ndarray:
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

            ts = np.linspace(0.0, 1.0, samples + 2, dtype=np.float32)[1:-1]
            xs_line = xi + ts * dx
            ys_line = yi + ts * dy
            rows, cols = rowcol(tfm, xs_line, ys_line, op=float)
            rows = np.clip(rows.round().astype(int), 0, cost.shape[0] - 1)
            cols = np.clip(cols.round().astype(int), 0, cost.shape[1] - 1)
            mean_cost = cost[rows, cols].mean()
            D[i, j] = D[j, i] = xy_dist * (1.0 + weight * mean_cost)

    return D


def make_canonical_seeds(chm_seeds, den_seeds, chm_raster, cost_surface, out_path,
                         eps_scale=0.4, min_eps=2, max_eps=8, z_thresh=-1, min_samples=2,
                         merge_radius=1.5, cost_weight=0.5, xy_thresh=0.8, dz_merge=0,
                         keep_all_stage1=True, stage1_top=1, max_per_cluster=0,
                         nms_base=0, nms_scale=0, debug_dist=True, keep=None, nodata_cost=1):

    if keep is None:
        keep = ["geometry", "height", "origin"]
    chm = gpd.read_file(chm_seeds); chm["origin"] = "chm"
    den = gpd.read_file(den_seeds); den["origin"] = "density"
    chm.rename(columns={"ch_max": "height"}, inplace=True)
    den.rename(columns={"den_max": "height"}, inplace=True)
    if "height" not in chm: chm = _add_chm_height(chm, chm_raster)
    if "height" not in den: den = _add_chm_height(den, chm_raster)

    seeds = gpd.GeoDataFrame(pd.concat([chm[keep], den[keep]], ignore_index=True),
                             geometry="geometry", crs=chm.crs)
    if seeds.empty:
        click.echo("No seeds after CHM sampling.", err=True)
        sys.exit(1)

    seeds["x"], seeds["y"] = seeds.geometry.x, seeds.geometry.y
    pts_xy = seeds[["x", "y"]].values
    tree = cKDTree(pts_xy)

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

    with rasterio.open(cost_surface) as src:
        cost_arr = src.read(1).astype(np.float32)
        cost_arr[cost_arr == src.nodatavals[0]] = nodata_cost
        cost_tfm = src.transform

    xs, ys = stage1["x"].values, stage1["y"].values
    D = _build_distance_matrix(xs, ys, cost_arr, cost_tfm,
                              cost_weight, xy_thresh, samples=12)

    if debug_dist:
        dvals = D[np.triu_indices(len(D), 1)]
        click.echo(f"d_eff  min/median/max = "
                   f"{dvals.min():.2f} / {np.median(dvals):.2f} / {dvals.max():.2f}")

    db = DBSCAN(eps=merge_radius, min_samples=1,
                metric="precomputed").fit(D)
    stage1["cluster"] = db.labels_

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

    def _trim(df):
        if max_per_cluster <= 0 or len(df) <= max_per_cluster:
            return df
        return df.nlargest(max_per_cluster, "height")

    trimmed = (stage1.groupby("cluster", group_keys=False)
               .apply(_trim).reset_index(drop=True))

    final = _nms_per_crown(trimmed, nms_base, nms_scale)

    final = final.rename(columns={"height": "ch_max"})
    final.insert(0, "id", range(len(final)))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

    final[["id", "cluster", "ch_max", "origin", "geometry"]].to_file(
        out_path, layer="canonical_seeds", driver="GPKG", overwrite=True)

    click.echo(f"✓ canonical seeds: {len(final):,}  →  {out_path}")

