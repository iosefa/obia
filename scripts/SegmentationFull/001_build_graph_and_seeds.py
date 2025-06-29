#!/usr/bin/env python3
"""
build_graph_and_seeds.py
────────────────────────────────────────────────────────────────────
Creates
  • segments_graph.pkl  – NetworkX graph (nodes = SLIC polygons)
  • seeds.json          – DF / CHM / PC / FILL seeds {seg_id: {...}}

Now includes an adaptive “gap-fill” step, so even if DeepForest is sparse
every canopy patch still gets a seed.
"""

import json, math, pickle, uuid, warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Point
from scipy.ndimage import maximum_filter, gaussian_filter
import skimage.feature                            #   blob_log

# ────────── paths ──────────────────────────────────────────────────
SEGMENTS_GPKG = Path("../docs/example_data/slic_segments_full.tif/segments.gpkg")
DF_BOXES      = Path("../docs/example_data/detections_pruned.gpkg")      # can be empty
CHM_RASTER    = Path("../docs/example_data/chm_full.tif")
WV3_9BAND     = "../docs/example_data/image_full.tif"
DEN_TIF       = "../docs/example_data/density.tif"

GRAPH_OUT = Path("segments_graph.pkl")
SEEDS_OUT = Path("seeds.json")

# ────────── parameters you might tune ──────────────────────────────
PEAK_H_MIN   = 2.5        # m  – CHM local-max threshold
PEAK_DIST_PX = 2          # px – local-max neighbourhood
CONF_DF  = 0.8
CONF_CHM = 0.9
CONF_PC  = 0.9
CONF_FILL = 0.7           # confidence for gap-fill seeds

SIG_RGB = 0.30            # colour similarity σ
SIG_H   = 1.50            # height similarity σ
MIN_BORDER_PX = 10        # ignore borders thinner than this (px)

GAP_MAX = 20.0            # m – no canopy centroid > GAP_MAX from a seed
MIN_CHM_SEEDS = 0.10      # we try to get >10 % of segments hit by CHM peaks
MIN_PC_SEEDS  = 0.05      # …and >5 % hit by PC density peaks
# ------------------------------------------------------------------


def mean_rgb_per_segment(gdf, multiband_path) -> np.ndarray:
    """Return an (N×3) array of mean R,G,B for each segment."""
    out = np.zeros((len(gdf), 3), dtype=float)
    with rasterio.open(multiband_path) as src:
        for bidx, col in zip((5, 3, 2), range(3)):      # WV-3: B5=R, B3=G, B2=B
            arr = src.read(bidx, masked=True)
            zs  = zonal_stats(gdf, arr, affine=src.transform,
                              stats="mean", nodata=0, n_jobs=-1)
            out[:, col] = [d["mean"] for d in zs]
    return out


def local_maxima(chm, h_min, size_px):
    """Return (row,col) indices of local maxima in `chm` ≥ h_min."""
    blur  = gaussian_filter(chm, sigma=1)
    peaks = np.logical_and(blur >= h_min,
                           blur == maximum_filter(blur, size=size_px))
    return np.column_stack(np.where(peaks))


# ───────────────────────────── main ────────────────────────────────
def main():
    # 1 ▸ load data --------------------------------------------------
    segs = gpd.read_file(SEGMENTS_GPKG).reset_index(drop=True)
    if "seg_id" not in segs.columns:
        segs["seg_id"] = np.arange(len(segs))
    segs.set_index("seg_id", inplace=True)

    df_boxes = gpd.read_file(DF_BOXES) if DF_BOXES.exists() else gpd.GeoDataFrame()

    # 2 ▸ node attributes -------------------------------------------
    print("• computing per-segment CHM mean & RGB mean …")
    segs["h_mean"] = [d["mean"] for d in zonal_stats(
        segs, CHM_RASTER, stats="mean", nodata=0, n_jobs=-1)]
    segs[["r", "g", "b"]] = mean_rgb_per_segment(segs, WV3_9BAND)

    with rasterio.open(CHM_RASTER) as src_chm:
        PIX = src_chm.res[0]   # spatial resolution (m)

    # 3 ▸ adjacency graph  ------------------------------------------
    print("• building adjacency graph …")
    G = nx.Graph()
    for sid, row in segs.iterrows():
        G.add_node(int(sid), h=row.h_mean, rgb=row[["r", "g", "b"]].tolist())

    sindex = segs.sindex
    for sid, geom in segs.geometry.items():
        for nbr in sindex.query(geom, predicate="touches"):
            if sid >= nbr:
                continue
            shared = geom.intersection(segs.geometry[nbr]).length
            if shared < MIN_BORDER_PX * PIX:
                continue
            v1, v2 = np.asarray(G.nodes[sid]["rgb"]), np.asarray(G.nodes[nbr]["rgb"])
            Δrgb = 1 - (np.dot(v1, v2) /
                        (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            Δh   = abs(G.nodes[sid]["h"] - G.nodes[nbr]["h"])
            sim  = math.exp(-(Δrgb / SIG_RGB) ** 2) * math.exp(-(Δh / SIG_H) ** 2)
            if sim > 0.2:
                G.add_edge(int(sid), int(nbr), sim=sim)

    # 4 ▸ DF seeds ---------------------------------------------------
    print("• mapping DeepForest boxes → seeds …")
    seeds = {}
    if not df_boxes.empty:
        points = gpd.GeoDataFrame({"seg_id": segs.index},
                                  geometry=segs.geometry.centroid, crs=segs.crs)
        df_matches = gpd.sjoin(points, df_boxes[["geometry"]],
                               predicate="within", how="inner").reset_index(drop=True)
        for i, r in df_matches.iterrows():
            seeds[int(r.seg_id)] = {"tree_id": f"DF_{i:06d}", "conf": CONF_DF}
    print(f"  DF seeds: {len(seeds):,}")

    # 5 ▸ CHM-peak seeds (with adaptive relax) -----------------------
    with rasterio.open(CHM_RASTER) as src:
        chm = src.read(1, masked=True).filled(0)
    chm[chm < PEAK_H_MIN] = 0
    peaks = local_maxima(chm, PEAK_H_MIN, PEAK_DIST_PX)
    xs, ys = rasterio.transform.xy(src.transform, peaks[:, 0], peaks[:, 1])
    peak_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=src.crs)
    if peak_gdf.crs != segs.crs:
        peak_gdf = peak_gdf.to_crs(segs.crs)

    def vector_join(points_gdf):
        if points_gdf.empty:
            return gpd.GeoDataFrame()
        joined = gpd.sjoin(points_gdf, segs[["geometry"]],
                           predicate="within", how="inner")
        # find correct *_right column
        rcol = next(c for c in joined.columns if c.endswith("_right"))
        return joined.drop_duplicates(subset=rcol).rename(columns={rcol: "seg_id"})

    peak_seg = vector_join(peak_gdf)
    # adaptive: try to relax h_min if we hit too few segments
    relax_iter = 0
    while len(peak_seg) < MIN_CHM_SEEDS * len(segs) and relax_iter < 3:
        relax_iter += 1
        print(f"  [CHM] too few peaks ({len(peak_seg)}); relaxing h_min ↓")
        chm_mask = chm.copy()
        chm_mask[chm_mask < (PEAK_H_MIN * 0.5 ** relax_iter)] = 0
        peaks = local_maxima(chm_mask, PEAK_H_MIN * 0.5 ** relax_iter,
                             PEAK_DIST_PX + relax_iter)
        xs, ys = rasterio.transform.xy(src.transform, peaks[:, 0], peaks[:, 1])
        peak_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=src.crs)
        if peak_gdf.crs != segs.crs:
            peak_gdf = peak_gdf.to_crs(segs.crs)
        peak_seg = vector_join(peak_gdf)

    added_chm = 0
    for _, r in peak_seg.iterrows():
        sid = int(r.seg_id)
        if sid not in seeds:
            seeds[sid] = {"tree_id": f"CHM_{uuid.uuid4().hex[:6]}", "conf": CONF_CHM}
            added_chm += 1
    print(f"  CHM seeds added: {added_chm:,}")

    # 6 ▸ PC-density seeds (adaptive) --------------------------------
    added_pc = 0
    if DEN_TIF.exists():
        with rasterio.open(DEN_TIF) as src:
            den = src.read(1, masked=True).filled(0)
        blobs = skimage.feature.blob_log(den, min_sigma=1, max_sigma=3,
                                         num_sigma=4, threshold=3)
        xs, ys = rasterio.transform.xy(src.transform, blobs[:, 0], blobs[:, 1])
        pc_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=src.crs)
        if pc_gdf.crs != segs.crs:
            pc_gdf = pc_gdf.to_crs(segs.crs)
        pc_seg = vector_join(pc_gdf)

        relax_iter = 0
        while len(pc_seg) < MIN_PC_SEEDS * len(segs) and relax_iter < 3:
            relax_iter += 1
            blobs = skimage.feature.blob_log(
                den, min_sigma=1, max_sigma=4 + relax_iter,
                num_sigma=6 + relax_iter, threshold=max(1, 3 - relax_iter))
            xs, ys = rasterio.transform.xy(src.transform, blobs[:, 0], blobs[:, 1])
            pc_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=src.crs)
            if pc_gdf.crs != segs.crs:
                pc_gdf = pc_gdf.to_crs(segs.crs)
            pc_seg = vector_join(pc_gdf)

        for _, r in pc_seg.iterrows():
            sid = int(r.seg_id)
            if sid not in seeds:
                seeds[sid] = {"tree_id": f"PC_{uuid.uuid4().hex[:6]}", "conf": CONF_PC}
                added_pc += 1
    print(f"  PC seeds added:  {added_pc:,}")

    # 7 ▸ gap-fill seeds --------------------------------------------
    print(f"• gap-fill: ensuring every centroid ≤ {GAP_MAX} m from a seed")
    seed_centroids = [segs.geometry.loc[s].centroid for s in seeds.keys()]
    tree = None
    try:
        from shapely.strtree import STRtree
        tree = STRtree(seed_centroids)
    except Exception:
        warnings.warn("STRtree unavailable; gap-fill skipped.")
    added_fill = 0
    if tree is not None:
        for sid, geom in segs.geometry.items():
            if sid in seeds:
                continue
            c = geom.centroid
            nearest = tree.nearest(c)
            if c.distance(nearest) > GAP_MAX:
                seeds[sid] = {
                    "tree_id": f"FILL_{uuid.uuid4().hex[:6]}",
                    "conf": CONF_FILL,
                }
                seed_centroids.append(c)
                tree = STRtree(seed_centroids)   # refresh
                added_fill += 1
    print(f"  FILL seeds added: {added_fill:,}")

    # summary --------------------------------------------------------
    counts = {k: 0 for k in ("DF", "CHM", "PC", "FILL")}
    for s in seeds.values():
        for k in counts:
            if s["tree_id"].startswith(k):
                counts[k] += 1
                break
    total = len(seeds)
    print(f"Total seeds: {total:,}  "
          f'DF={counts["DF"]:,}  CHM={counts["CHM"]:,}  '
          f'PC={counts["PC"]:,}  FILL={counts["FILL"]:,}')

    # 8 ▸ save artefacts --------------------------------------------
    with open(GRAPH_OUT, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(SEEDS_OUT, "w") as f:
        json.dump(seeds, f, indent=2)

    print("✓ graph  →", GRAPH_OUT)
    print("✓ seeds  →", SEEDS_OUT)


if __name__ == "__main__":
    main()