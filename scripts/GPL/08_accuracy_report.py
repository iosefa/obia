#!/usr/bin/env python3
"""
08_accuracy_report.py
────────────────────────────────────────────────────────────
Evaluate crown-level predictions.

• joins predictions with ground-truth crowns if needed
• writes:
    confusion.png      – confusion-matrix heat-map
    metrics.csv        – precision / recall / F1 per class
    calibration.png    – reliability diagram
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.calibration import calibration_curve

# ───────────────────────── paths ──────────────────────────────────
BASE          = Path("/Users/iosefa/repos/obia/docs/example_data/site_1")
PRED_GPKG     = BASE / "crowns_pred.gpkg"      # from step 07
LABEL_GPKG    = BASE / "crowns_labeled.gpkg"   # from step 01  (seg_id + feature_class)
CONF_PNG      = BASE / "confusion.png"
METRICS_CSV   = BASE / "metrics.csv"
CAL_PNG       = BASE / "calibration.png"

# column names in your layers --------------------------------------
ID_COL   = "segment_id"        # primary key
LABEL_COL= "feature_class"     # ground-truth species label


# ───────────────────────── load tables ────────────────────────────
pred = gpd.read_file(PRED_GPKG)[[ID_COL, "pred_class_id", "post_prob"]]
if LABEL_COL in pred.columns:
    crowns = pred.copy()
else:
    labels = gpd.read_file(LABEL_GPKG)[[ID_COL, LABEL_COL]]
    crowns = pred.merge(labels, on=ID_COL, how="inner", validate="1:1")

print(f"Evaluation crowns: {len(crowns):,}")

# ───── class-ID → human label map (numeric → original GT label) ───
unique_gt   = sorted(crowns[LABEL_COL].dropna().unique())
id2name     = {i: str(lbl) for i, lbl in enumerate(unique_gt)}
crowns["pred_lbl"] = crowns["pred_class_id"].map(
    lambda i: id2name.get(i, f"Unknown_{i}")
)

# Cast *all* labels to str so sklearn sees one dtype ----------------
crowns[LABEL_COL] = crowns[LABEL_COL].astype(str)
crowns["pred_lbl"] = crowns["pred_lbl"].astype(str)
class_names = sorted(
    pd.unique(crowns[LABEL_COL].tolist() + crowns["pred_lbl"].tolist())
)

# ─────────────────────── confusion matrix ─────────────────────────
cm = confusion_matrix(
    crowns[LABEL_COL], crowns["pred_lbl"], labels=class_names
)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig(CONF_PNG, dpi=300)
print(f"✓ confusion matrix → {CONF_PNG}")

# ───────────── precision / recall / F1 table ──────────────────────
prec, rec, f1, support = precision_recall_fscore_support(
    crowns[LABEL_COL], crowns["pred_lbl"], labels=class_names, zero_division=0
)
metrics = pd.DataFrame(
    dict(
        class_name=class_names,
        precision=np.round(prec, 3),
        recall=np.round(rec, 3),
        f1=np.round(f1, 3),
        support=support,
    )
)
ovr = (
    metrics[["precision", "recall", "f1"]]
    .multiply(metrics["support"], axis=0)
    .sum()
    / metrics["support"].sum()
)
metrics.loc[len(metrics)] = ["OVERALL", *ovr.round(3), metrics["support"].sum()]
metrics.to_csv(METRICS_CSV, index=False)
print(f"✓ metrics table    → {METRICS_CSV}")

# ───────────────────── reliability diagram ────────────────────────
probs = crowns["post_prob"].values
truth = (crowns[LABEL_COL] == crowns["pred_lbl"]).astype(int).values
bin_acc, bin_prob = calibration_curve(truth, probs, n_bins=10, strategy="uniform")

plt.figure(figsize=(4, 4))
plt.plot(bin_prob, bin_acc, "o-", label="model")
plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
plt.xlabel("Predicted probability")
plt.ylabel("Actual accuracy")
plt.title("Reliability diagram")
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(CAL_PNG, dpi=300)
print(f"✓ calibration plot → {CAL_PNG}")