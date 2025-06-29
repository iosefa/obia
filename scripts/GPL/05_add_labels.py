#!/usr/bin/env python3
"""
05_add_labels.py
────────────────────────────────────────────────────────────
Add the missing `feature_class` column to crown_features.parquet
by joining the original labelled-crown layer on `seg_id`.

Run only once (or whenever you regenerate crown_features.parquet).

Inputs
------
crowns_labeled.gpkg        ← from 01_labeled_crowns.py   (has seg_id + feature_class)
crown_features.parquet     ← from 04_build_crown_graph.py (missing feature_class)

Outputs
-------
crown_features.parquet     ← OVERWRITTEN with the label column added
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd

# paths ────────────────────────────────────────────────────────────
BASE_DIR        = Path("/Users/iosefa/repos/obia/docs/example_data/site_1")
LABEL_GPKG      = BASE_DIR / "crowns_labeled.gpkg"
FEATURES_PARQ   = BASE_DIR / "crown_features.parquet"   # produced in step 04

# 1 ▸ read data -----------------------------------------------------
print("• reading labelled crowns …")
labels_gdf = gpd.read_file(LABEL_GPKG)[["segment_id", "feature_class"]]

print("• reading feature table …")
feat_df = pd.read_parquet(FEATURES_PARQ)

# 2 ▸ merge on seg_id ----------------------------------------------
merged = feat_df.merge(labels_gdf, on="segment_id", how="left")

n_missing = merged["feature_class"].isna().sum()
if n_missing:
    print(f"⚠  {n_missing} crowns still lack a label (left as NaN)")

# 3 ▸ overwrite parquet --------------------------------------------
merged.to_parquet(FEATURES_PARQ, index=False)
print(f"✓ feature table updated → {FEATURES_PARQ}")
