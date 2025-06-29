#!/usr/bin/env python3
"""
06_train_gcn.py
────────────────────────────────────────────────────────────
Train a 2-layer GraphSAGE on the crown graph.

Inputs  (produced in steps 04–05)
---------------------------------
crown_graph.pkl      – pickled NetworkX graph   (edge attrs: sim, dist, dH)
node_features.pt     – torch.FloatTensor [N × F]
labels.pt            – torch.LongTensor  [N × C]  (one-hot, 0-rows = unlabeled)

Outputs
-------
gcn_best.pt          – best-epoch PyTorch model
probs.npy            – softmax posteriors  (N × C)
train_val_idx.json   – indices for reproducibility
"""

# ───────────────────────────── USER PATHS ──────────────────────────
BASE        = "/Users/iosefa/repos/obia/docs/example_data/site_1"
GRAPH_PKL   = f"{BASE}/crown_graph.pkl"
NODE_FEATS  = f"{BASE}/node_features.pt"
LABELS_PT   = f"{BASE}/labels.pt"

MODEL_OUT   = f"{BASE}/gcn_best.pt"
PROBS_NPY   = f"{BASE}/probs.npy"
SPLIT_JSON  = f"{BASE}/train_val_idx.json"
# -------------------------------------------------------------------

import json, pickle, random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

# reproducibility ───────────────────────────────────────────────────
SEED   = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 2048        # node-wise batch for NeighborLoader
EPOCHS = 200
PATIENCE = 20
LR     = 2e-3
HID    = 256
DROPOUT= 0.30
VAL_FRAC = 0.20      # 80/20 split of labelled nodes

# ─────────────────────── 1 ▸ load artefacts ────────────────────────
print("• loading crown graph …")
with open(GRAPH_PKL, "rb") as f:
    G_nx: nx.Graph = pickle.load(f)

print("• loading tensors …")
x_all = torch.load(NODE_FEATS)      # [N, F]
y_all = torch.load(LABELS_PT)       # [N, C]

N, F = x_all.shape
C     = y_all.shape[1]
label_mask = y_all.sum(dim=1) > 0   # bool: True = has label

# ───────────────────── 2 ▸ split train / val ───────────────────────
lab_idx = torch.where(label_mask)[0].cpu().numpy()
y_arg   = y_all[label_mask].argmax(dim=1).cpu().numpy()
tr_idx, val_idx = train_test_split(
    lab_idx,
    test_size=VAL_FRAC,
    stratify=y_arg,
    random_state=SEED,
)

with open(SPLIT_JSON, "w") as fp:
    json.dump({"train": tr_idx.tolist(), "val": val_idx.tolist()}, fp, indent=2)

# masks (BoolTensor of size N)
train_mask = torch.zeros(N, dtype=torch.bool); train_mask[tr_idx] = True
val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[val_idx] = True

# ───────────────────── 3 ▸ pyg Data object ─────────────────────────
data = from_networkx(G_nx)
data.x = x_all
data.y = y_all.argmax(dim=1)        # CrossEntropy needs class index
data.train_mask = train_mask
data.val_mask   = val_mask
data = data.to(DEVICE)

# ───────────────────── 4 ▸ neighbor sampler (mini-batch) ───────────
loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],      # 2 hops
    batch_size=BATCH,
    input_nodes=None,            # sample all nodes each epoch
)

# ───────────────────── 5 ▸ model & optimiser ───────────────────────
class CrownSAGE(nn.Module):
    def __init__(self, fin, hid, fout, p=0.3):
        super().__init__()
        self.conv1 = SAGEConv(fin, hid)
        self.conv2 = SAGEConv(hid, fout)
        self.drop  = nn.Dropout(p)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x

model = CrownSAGE(F, HID, C, DROPOUT).to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion  = nn.CrossEntropyLoss()

# ───────────────────── 6 ▸ training loop with early stop ───────────
best_val  = 0.0
pat_cnt   = 0

for epoch in range(1, EPOCHS+1):
    # ---- train ----------------------------------------------------
    model.train(); tot_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimiser.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward(); optimiser.step()
        tot_loss += loss.item() * batch.train_mask.sum().item()
    tot_loss /= train_mask.sum().item()

    # ---- validation ----------------------------------------------
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out[val_mask].argmax(dim=1).cpu()
        truth = data.y[val_mask].cpu()
        val_acc = accuracy_score(truth, preds)
        val_f1  = f1_score(truth, preds, average="macro")

    print(f"Epoch {epoch:03d}  loss={tot_loss:.4f}  val-F1={val_f1:.3f}")

    # ---- early stopping ------------------------------------------
    if val_f1 > best_val + 1e-4:
        best_val = val_f1
        pat_cnt  = 0
        torch.save(model.state_dict(), MODEL_OUT)
    else:
        pat_cnt += 1
        if pat_cnt >= PATIENCE:
            print("Early stop ✓")
            break

print(f"Best val-F1 = {best_val:.3f}  → model saved to {MODEL_OUT}")

# ───────────────────── 7 ▸ posteriors for *all* crowns ─────────────
model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index).cpu().numpy()
probs = np.exp(logits - logits.max(axis=1, keepdims=True))
probs /= probs.sum(axis=1, keepdims=True)
np.save(PROBS_NPY, probs)
print(f"✓ posteriors → {PROBS_NPY}  (shape {probs.shape})")