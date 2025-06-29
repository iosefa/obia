#!/usr/bin/env python3
"""
04_build_crown_graph.py
────────────────────────────────────────────────────────────
Create a context-aware crown graph that joins manual OBIA
descriptors with CNN embeddings.

Inputs
------
1. objects_manual_feats.gpkg  – output of 02_manual_feats.py
2. cnn_embeddings.npy         – output of 03_train_cnn_embedder.py

Outputs
-------
• crown_graph.pkl        – pickled NetworkX graph   (via the std-lib pickle)
• crown_features.parquet – node-attribute table
• graph_qc.gpkg          – OPTIONAL GIS layer for QC
"""

# ───────────────────────────── USER PATHS ──────────────────────────
GPKG_IN   = "/Users/iosefa/repos/obia/docs/example_data/site_1/crowns_manual_feats.gpkg"
EMB_NPY   = "/Users/iosefa/repos/obia/docs/example_data/site_1/crown_embeddings.npy"
GRAPH_PKL = "/Users/iosefa/repos/obia/docs/example_data/site_1/crown_graph.pkl"
FEATURES_OUT = "/Users/iosefa/repos/obia/docs/example_data/site_1/crown_features.parquet"
QC_GPKG  = "/Users/iosefa/repos/obia/docs/example_data/site_1/graph_qc.gpkg"   # ← set None to skip
# -------------------------------------------------------------------

import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from shapely.geometry import Point

# ─────────────── parameters you may wish to tune ───────────────────
K_NEIGHBORS = 8        # fallback # nearest crowns per node
R_MAX       = 25.0     # metres – radius search for “natural” neighbours
SIM_MIN     = 0.20     # edges below this cosine sim are discarded
CNN_KEEP    = 20       # keep first N dims of the 128-D vector
# -------------------------------------------------------------------


def main() -> None:
    # ──────────────── load crowns & CNN embeddings ─────────────────
    gdf = gpd.read_file(GPKG_IN)
    if "seg_id" not in gdf.columns:
        gdf["seg_id"] = np.arange(len(gdf))

    cnn = np.load(EMB_NPY)
    if cnn.shape[0] != len(gdf):
        raise ValueError(
            f"Embed rows ({cnn.shape[0]}) ≠ crowns ({len(gdf)}). "
            "Check that GPKG and NPY come from the same site."
        )

    # attach embeddings as separate columns
    for i in range(cnn.shape[1]):
        gdf[f"cnn_{i:03d}"] = cnn[:, i]

    # ───────────── pick features used for edge similarity ──────────
    numeric_cols = (
        [c for c in gdf.columns if c.startswith(("spectral_", "texture_", "h_", "ch", "pai", "fhd"))]
        + [f"cnn_{i:03d}" for i in range(CNN_KEEP)]
    )
    feat_mat = gdf[numeric_cols].to_numpy(dtype=np.float32)

    # ───────────── KD-tree on centroids for neighbour search ───────
    gdf["centroid"] = gdf.geometry.centroid
    coords = np.column_stack([gdf.centroid.x, gdf.centroid.y])

    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        raise SystemExit("✗ scikit-learn missing.  pip install scikit-learn")

    kdt = KDTree(coords, metric="euclidean")

    # ───────────── build graph & attach node attributes ─────────────
    G = nx.Graph()
    for idx, seg_id in enumerate(gdf.seg_id):
        G.add_node(
            int(seg_id),
            **{k: float(gdf.iloc[idx][k]) for k in numeric_cols},
        )

    for idx, seg_id in enumerate(gdf.seg_id):
        # radius search
        r_idx = kdt.query_radius(coords[idx : idx + 1], r=R_MAX, return_distance=False)[0]

        # ensure at least K neighbours
        if len(r_idx) < K_NEIGHBORS + 1:  # +1 for self
            k_idx = kdt.query(coords[idx : idx + 1], k=K_NEIGHBORS + 1, return_distance=False)[0]
            neigh_idx = np.unique(np.concatenate([r_idx, k_idx]))
        else:
            neigh_idx = r_idx
        neigh_idx = neigh_idx[neigh_idx != idx]  # drop self

        sims = cosine_similarity(
            feat_mat[idx : idx + 1],    # (1, d)
            feat_mat[neigh_idx],        # (m, d)
        )[0]

        for n_i, sim in zip(neigh_idx, sims):
            if sim < SIM_MIN:
                continue
            u = int(seg_id)
            v = int(gdf.seg_id.iloc[n_i])
            if G.has_edge(u, v):
                if sim > G.edges[u, v]["sim"]:
                    G.edges[u, v]["sim"] = float(sim)
            else:
                dist = float(np.linalg.norm(coords[idx] - coords[n_i]))
                Δh = float(
                    abs(
                        gdf.get("ch", pd.Series(np.nan, index=gdf.index)).iloc[idx]
                        - gdf.get("ch", pd.Series(np.nan, index=gdf.index)).iloc[n_i]
                    )
                )
                G.add_edge(u, v, sim=float(sim), dist=dist, dH=Δh)

    # ───────────── save artefacts ───────────────────────────────────
    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ graph → {GRAPH_PKL}  (|V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,})")

    gdf.drop(columns=["centroid"]).to_parquet(FEATURES_OUT, index=False)
    print(f"✓ features → {FEATURES_OUT}")

    # optional QC layer: nodes (centroids) + edge lines -------------
    if QC_GPKG:
        nodes = gpd.GeoDataFrame({"seg_id": gdf.seg_id}, geometry=gdf.centroid, crs=gdf.crs)

        edges_geo = []
        for u, v, d in G.edges(data=True):
            p1 = nodes.loc[nodes.seg_id == u, "geometry"].values[0]
            p2 = nodes.loc[nodes.seg_id == v, "geometry"].values[0]
            edges_geo.append({"geometry": gpd.GeoSeries([p1, p2]).unary_union, "sim": d["sim"]})
        edges_gdf = gpd.GeoDataFrame(edges_geo, crs=gdf.crs)

        nodes.to_file(QC_GPKG, layer="nodes", driver="GPKG", overwrite=True)
        edges_gdf.to_file(QC_GPKG, layer="edges", driver="GPKG", overwrite=True)
        print(f"✓ QC layers → {QC_GPKG}")


if __name__ == "__main__":
    main()