#!/usr/bin/env python3
"""
07_infer_full_map.py ― apply the trained GCN to every crown
────────────────────────────────────────────────────────────
* Re-build the 2-layer GraphSAGE used in 06_train_gcn.py
* Load its state-dict (gcn_best.pt) and run inference
* Append pred_class_id, post_prob, uncert_margin to crowns layer
"""

# ───────────────────────── USER PATHS ──────────────────────────────
BASE = "/Users/iosefa/repos/obia/docs/example_data/site_1"

GRAPH_PKL      = f"{BASE}/crown_graph.pkl"
X_PT           = f"{BASE}/node_features.pt"
Y_PT           = f"{BASE}/labels.pt"
MODEL_SD       = f"{BASE}/gcn_best.pt"

FEATURES_PARQ  = f"{BASE}/crown_features.parquet"
CROWNS_GPKG    = f"{BASE}/crowns_manual_feats.gpkg"  # has `segment_id`

OUT_GPKG       = f"{BASE}/crowns_pred.gpkg"
OUT_NPY        = f"{BASE}/probs.npy"                 # set None to skip
# ------------------------------------------------------------------

import pickle, warnings
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
import networkx as nx

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID_COL = "segment_id"        # <── fixed identifier name

# ─────────── GraphSAGE definition (same as step 06) ────────────────
class GraphSage2(torch.nn.Module):
    def __init__(self, in_ch: int, hid: int, n_cls: int):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid)
        self.conv2 = SAGEConv(hid, n_cls)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.conv1(x, ei).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, ei)

# ─────────────────── load graph & tensors ──────────────────────────
print("• loading graph & tensors …")
with open(GRAPH_PKL, "rb") as f:
    Gx: nx.Graph = pickle.load(f)

data         = from_networkx(Gx, group_node_attrs=None)
data.x       = torch.load(X_PT)
n_nodes, inF = data.x.shape
n_classes    = torch.load(Y_PT).shape[1]
data = data.to(DEVICE)
print(f"  nodes={n_nodes:,}  features={inF}  classes={n_classes}")

# ─────────────────── rebuild model & load weights ──────────────────
model = GraphSage2(inF, 256, n_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_SD, map_location=DEVICE))
model.eval()

# ────────────────────────── inference ──────────────────────────────
with torch.no_grad():
    logits = model(data)
probs   = F.softmax(logits, dim=1).cpu().numpy()
pred    = probs.argmax(axis=1)
top2    = np.partition(-probs, 1, axis=1)[:, :2]
uncert  = -top2[:, 0] - (-top2[:, 1])      # top-1 minus top-2
print("✓ inference done")

# ─────────────────────────── merge back ────────────────────────────
crowns   = gpd.read_file(CROWNS_GPKG)
feat_tbl = pd.read_parquet(FEATURES_PARQ)

if ID_COL not in crowns.columns or ID_COL not in feat_tbl.columns:
    raise SystemExit(f"✗ `{ID_COL}` must exist in both crowns and feature table")

assert len(feat_tbl) == len(pred) == len(crowns)

pred_df = pd.DataFrame(
    {
        ID_COL: feat_tbl[ID_COL].values,
        "pred_class_id": pred,
        "post_prob": probs.max(axis=1),
        "uncert_margin": uncert,
    }
)

crowns = crowns.merge(pred_df, on=ID_COL, how="left")
crowns.to_file(OUT_GPKG, driver="GPKG", overwrite=True)
print(f"✓ crowns with predictions → {OUT_GPKG}")

if OUT_NPY:
    np.save(OUT_NPY, probs)
    print(f"✓ posterior matrix saved → {OUT_NPY}  {probs.shape}")