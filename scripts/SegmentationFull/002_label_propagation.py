#!/usr/bin/env python3
"""
Propagate seed labels across the segment graph (CRF-style)
and write a table: seg_id ➜ tree_id.
"""

import json, pickle, networkx as nx
import numpy as np, pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

GRAPH_IN   = Path("segments_graph.pkl")
SEEDS_IN   = Path("seeds.json")
LABEL_CSV  = Path("segments_labeled.parquet")

ALPHA      = 0.3 # 0.2          # 0=stick to seed, 1=full smoothing
MAX_ITER   = 30 # 20
EPS        = 1e-9

# ── load artefacts ────────────────────────────────────────────────
G  = pickle.load(open(GRAPH_IN, "rb"))
seeds = {int(k): v for k, v in json.load(open(SEEDS_IN)).items()}

# initialise label probabilities
Y = {n: {} for n in G.nodes}
for n, s in seeds.items():
    Y[n] = {s["tree_id"]: 1.0}

# edge similarity cached
# for u, v, d in G.edges(data=True):
#     d["sim"] = 1 - d["weight"]             # 0‒1

# ── iterative propagation ────────────────────────────────────────
for _ in tqdm(range(MAX_ITER), desc="propagating"):
    Ynew = {}
    for n in G.nodes:
        if n in seeds:                         # hard seed
            Ynew[n] = Y[n]; continue
        agg = {}
        for nbr in G.neighbors(n):
            w = G[n][nbr]["sim"]
            for lbl, p in Y.get(nbr, {}).items():   # safe lookup
                agg[lbl] = agg.get(lbl, 0) + w * p
        if not agg:
            continue
        z = sum(agg.values()) + EPS
        Ynew[n] = {lbl: (1-ALPHA)*p/z for lbl, p in agg.items()}
    Y = Ynew

label_of = {n: max(Y[n], key=Y[n].get) for n in Y if Y[n]}
pd.DataFrame(label_of.items(), columns=["seg_id", "tree_id"]).to_parquet(LABEL_CSV, index=False)
print("✓ labels →", LABEL_CSV)
