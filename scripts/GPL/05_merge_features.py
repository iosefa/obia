#!/usr/bin/env python3
"""
05_merge_features.py  – SAFE version
─────────────────────────────────────────────────────────────────
* joins OBIA descriptors + CNN embeddings + simple graph metrics
* imputes NaNs with column medians
* removes constant columns
* z-scores the rest
"""

from pathlib import Path
import pickle, warnings, numpy as np, pandas as pd, torch, networkx as nx
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ───────── paths ─────────
BASE         = Path("/Users/iosefa/repos/obia/docs/example_data/site_1")
GRAPH_PKL    = BASE / "crown_graph.pkl"
FEATURES_IN  = BASE / "crown_features.parquet"

NODE_FEATS   = BASE / "node_features.pt"
LABELS_PT    = BASE / "labels.pt"
LOOKUP_CSV   = BASE / "feature_lookup.csv"

ID_COL       = "segment_id"          # <- edit if you renamed it
LABEL_COL    = "feature_class"       # <- edit if you renamed it
DROP_GEOM    = True
# ─────────────────────────

print("• reading artefacts …")
with open(GRAPH_PKL, "rb") as f:
    G: nx.Graph = pickle.load(f)
feats = pd.read_parquet(FEATURES_IN)
if DROP_GEOM and "geometry" in feats.columns:
    feats = feats.drop(columns="geometry")

# ---------- add simple graph metrics --------------------------------
deg_map  = dict(G.degree())
mean_sim = {n: (np.mean([d["sim"] for *_ , d in G.edges(n, data=True)])
                if G.degree(n) else 0.0)
            for n in G.nodes}

feats["deg"]      = feats[ID_COL].map(deg_map).astype("float32")
feats["mean_sim"] = feats[ID_COL].map(mean_sim).astype("float32")

# ---------- numeric matrix ------------------------------------------
num_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove(ID_COL)           # pure identifier
X_raw = feats[num_cols].to_numpy(dtype="float32")

# 1) impute NaNs with column median
X_imputed = SimpleImputer(strategy="median").fit_transform(X_raw)

# 2) drop constant columns (std == 0  after imputing)
std = X_imputed.std(axis=0)
keep_mask = std > 0
X_dropped = X_imputed[:, keep_mask]
kept_cols = [c for c, k in zip(num_cols, keep_mask) if k]

# 3) standardise
scaler = StandardScaler()
X = scaler.fit_transform(X_dropped).astype("float32")

# ---------- labels ---------------------------------------------------
y_raw = feats[LABEL_COL].astype("category").to_numpy().reshape(-1, 1)
enc   = OneHotEncoder(sparse_output=False, dtype=np.int64)
Y     = enc.fit_transform(y_raw)

# ---------- save -----------------------------------------------------
torch.save(torch.tensor(X), NODE_FEATS)
torch.save(torch.tensor(Y), LABELS_PT)
pd.Series(kept_cols).to_csv(LOOKUP_CSV, index=False, header=False)

print(f"✓ node features : {NODE_FEATS}   shape = {X.shape}")
print(f"✓ labels        : {LABELS_PT}    classes = {Y.shape[1]}")
print(f"✓ kept columns  : {len(kept_cols)}  (dropped {len(num_cols)-len(kept_cols)})")