#!/usr/bin/env python
# crown_segmentation.py
#
# Builds one crown polygon per seed‑cluster, combining probability, CHM and
# cost, and restricts the crown’s size and shape during region growing.
# ---------------------------------------------------------------------------

import re, warnings, math
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import geometry_window
from rasterstats import zonal_stats
from shapely.geometry import box
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import crown_config as cfg

# ---------------------------------------------------------------------------
_CLUSTER_RE = re.compile(r"seed_(\d+)", flags=re.IGNORECASE)
MAX_AREA    = math.pi * cfg.MAX_RADIUS ** 2          # m²
# ---------------------------------------------------------------------------


# ---------------------------  HELPERS  -------------------------------------
def list_prob_files_by_cluster(prob_dir: Path) -> Dict[int, List[Path]]:
    out: Dict[int, List[Path]] = {}
    for tif in prob_dir.glob("seed_*.tif"):
        m = _CLUSTER_RE.search(tif.name)
        if m:
            cid = int(m.group(1))
            out.setdefault(cid, []).append(tif)
    return out


def smooth_ignore_nodata(arr: np.ndarray, sigma: float) -> np.ndarray:
    nodata = np.isnan(arr)
    valid  = (~nodata).astype(float)
    blurred_data  = gaussian_filter(np.nan_to_num(arr), sigma=sigma)
    blurred_valid = gaussian_filter(valid,        sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = blurred_data / blurred_valid
    out[blurred_valid == 0] = np.nan
    return out


def load_slic_subset(slic_gdf, footprints):
    return slic_gdf[slic_gdf.intersects(unary_union(footprints))]


def neighboring_polys(gdf):
    sindex, labels = gdf.sindex, gdf.index
    neigh = {lab: set() for lab in labels}
    for pos_i, geom in enumerate(gdf.geometry):
        lab_i = labels[pos_i]
        for pos_j in sindex.query(geom, predicate="touches"):
            if pos_j == pos_i:
                continue
            lab_j = labels[pos_j]
            neigh[lab_i].add(lab_j)
            neigh[lab_j].add(lab_i)
    return neigh


def compactness(geom):
    """Perimeter² / (4 π Area); 1 = perfect circle."""
    if geom.area == 0:
        return np.inf
    return geom.length ** 2 / (4 * math.pi * geom.area)
# ---------------------------------------------------------------------------


def main():
    cfg.OUT_DIR.mkdir(exist_ok=True)

    slic_all  = gpd.read_file(cfg.SLIC_FILE)
    seeds_all = gpd.read_file(cfg.SEED_FILE)
    prob_map  = list_prob_files_by_cluster(cfg.PROB_DIR)
    cost_src  = rasterio.open(cfg.COST_FILE) if cfg.COST_MAX is not None else None

    for cluster_id in tqdm(sorted(seeds_all["cluster"].unique()), desc="crowns"):
        tif_list = prob_map.get(cluster_id, [])
        if not tif_list:
            warnings.warn(f"No probability rasters for cluster {cluster_id}")
            continue

        rasters, footprints = [], []

        # ───────── Build composite probability for every chip ───────────────
        for tif_path in tif_list:
            src = rasterio.open(tif_path)
            prob_raw = src.read(1, masked=True).filled(np.nan).astype("float32")
            prob_s   = smooth_ignore_nodata(prob_raw, cfg.GAUSSIAN_SIGMA)

            # CHM → prob grid
            with rasterio.open(cfg.CHM_FILE) as chm_src:
                chm_r = np.empty_like(prob_s, dtype="float32")
                reproject(
                    rasterio.band(chm_src, 1), chm_r,
                    src_transform=chm_src.transform, src_crs=chm_src.crs,
                    dst_transform=src.transform,  dst_crs=src.crs,
                    dst_nodata=np.nan, resampling=Resampling.bilinear
                )

            # cost → prob grid
            if cfg.COST_MAX is not None:
                cost_r = np.empty_like(prob_s, dtype="float32")
                reproject(
                    rasterio.band(cost_src, 1), cost_r,
                    src_transform=cost_src.transform, src_crs=cost_src.crs,
                    dst_transform=src.transform,  dst_crs=src.crs,
                    dst_nodata=np.nan, resampling=Resampling.bilinear
                )
            else:
                cost_r = np.zeros_like(prob_s, dtype="float32")

            # normalise
            def _norm(a):
                if np.all(np.isnan(a)):
                    return np.zeros_like(a)
                vmin, vmax = np.nanpercentile(a, (5, 95))
                return np.clip((a - vmin) / (vmax - vmin + 1e-6), 0, 1)

            composite = (
                  cfg.W_PROB * prob_s
                + cfg.W_CHM  * _norm(chm_r)
                + cfg.W_COST * (1.0 - _norm(cost_r))
            )
            composite = smooth_ignore_nodata(composite, sigma=1)

            rasters.append((composite, src.transform, src.crs))
            footprints.append(box(*src.bounds))
            src.close()
        # ────────────────────────────────────────────────────────────────────

        slic_sub = load_slic_subset(slic_all, footprints)
        if slic_sub.empty:
            warnings.warn(f"No SLIC polygons intersect cluster {cluster_id}")
            continue
        slic_sub = slic_sub.copy()

        seeds_geom = seeds_all.loc[seeds_all["cluster"] == cluster_id, "geometry"].union_all()
        start_idxs = slic_sub[slic_sub.intersects(seeds_geom)].index.tolist()
        if not start_idxs:
            warnings.warn(f"No seed polygons found for cluster {cluster_id}")
            continue

        neigh = neighboring_polys(slic_sub)

        # polygon‑wise Pmax
        Pmax = np.full(len(slic_sub), -np.inf)
        for arr, aff, _ in rasters:
            zs = zonal_stats(slic_sub.geometry, arr, affine=aff,
                             nodata=np.nan, stats=("max",))
            Pmax = np.maximum(Pmax, [d["max"] if d["max"] is not None else -np.inf for d in zs])
        slic_sub["Pmax"] = Pmax

        # polygon‑wise mean cost
        if cost_src:
            bb = unary_union(footprints)
            win = geometry_window(cost_src, [bb])
            cost_big = cost_src.read(1, window=win, masked=True).filled(np.nan)
            cost_aff = cost_src.window_transform(win)
            zs_c = zonal_stats(slic_sub.geometry, cost_big, affine=cost_aff,
                               nodata=np.nan, stats=("mean",))
            slic_sub["Cmean"] = [d["mean"] for d in zs_c]
        else:
            slic_sub["Cmean"] = 0

        # ───────────── region growing with geometry constraints ─────────────
        selected, frontier = set(start_idxs), set(start_idxs)
        crown_geom = unary_union(slic_sub.loc[list(selected), "geometry"])

        while frontier:
            current = frontier.pop()
            for nb in neigh[current] - selected:
                row = slic_sub.loc[nb]

                # attribute tests
                if row.Pmax < cfg.PROB_THRESHOLD:
                    continue
                if cfg.COST_MAX is not None and row.Cmean > cfg.COST_MAX:
                    continue
                shared = row.geometry.boundary.intersection(crown_geom.boundary).length
                if shared / row.geometry.length < cfg.EDGE_MIN_FRAC:
                    continue

                # geometric tests (area & compactness) on the *candidate* crown
                candidate = crown_geom.union(row.geometry)
                if candidate.area > MAX_AREA:
                    continue
                if compactness(candidate) > cfg.COMPACTNESS_MAX:
                    continue

                # accept neighbour
                selected.add(nb)
                frontier.add(nb)
                crown_geom = candidate  # update union geom

        if not selected:
            warnings.warn(f"Crown {cluster_id} grew to zero polygons — skipped")
            continue

        out_gdf = gpd.GeoDataFrame(
            {"cluster": [cluster_id], "n_parts": [len(selected)]},
            geometry=[crown_geom], crs=slic_sub.crs
        )
        out_path = cfg.OUT_DIR / f"crown_{cluster_id:04d}.gpkg"
        out_gdf.to_file(out_path, driver="GPKG")
        print(f"✓ wrote {out_path}")

    if cost_src:
        cost_src.close()


if __name__ == "__main__":
    main()
