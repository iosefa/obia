#!/usr/bin/env python3
"""
01_labeled_crowns.py
────────────────────────────────────────────────────────────────────
Assigns species labels to pre-delineated crown polygons using ground
training points.  Crowns touched by two or more classes are flagged as
“mixed” and *excluded* from the labelled output.

Inputs
------
/docs/example_data/site_1/merged_crowns.gpkg   : crowns (must contain a unique `segment_id`)
/docs/example_data/site_1/training.gpkg        : point samples with a `class` field

Outputs
-------
crowns_labeled.gpkg   : clean crowns with a `feature_class` column
mixed_segments.csv    : list of ambiguous segment_id values (for QA)

Author: <you>
Date  : 2025-05-31
"""

from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import pandas as pd

# ------------------------------------------------------------------
# helper – unchanged from your snippet
# ------------------------------------------------------------------
def label_segments(
    segments: gpd.GeoDataFrame,
    labelled_points: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Attach a `feature_class` attribute to each segment whose polygon
    intersects points of exactly one training class.

    Returns
    -------
    labelled_segments : GeoDataFrame  (only uniquely labelled crowns)
    mixed_segments    : list[str]     (segment_id values that are mixed)
    """
    mixed_segments = []
    labelled_segments = segments.copy()

    intersections = gpd.sjoin(
        labelled_segments, labelled_points, how="inner", predicate="intersects"
    )

    # iterate over each crown polygon that intersected ≥1 point
    for poly_idx, grp in intersections.groupby(intersections.index):
        classes = grp["class"].unique()
        if len(classes) == 1:
            labelled_segments.loc[poly_idx, "feature_class"] = classes[0]
        else:
            segment_id = grp["segment_id"].values[0]
            mixed_segments.append(segment_id)

    labelled_segments = labelled_segments[labelled_segments["feature_class"].notna()]
    return labelled_segments, mixed_segments


# ------------------------------------------------------------------
# paths – adjust if your directory structure changes
# ------------------------------------------------------------------
BASE = Path("/Users/iosefa/repos/obia/docs/example_data/site_1")
CROWNS_GPKG   = BASE / "merged_crowns.gpkg"
TRAIN_GPKG    = BASE / "training.gpkg"

OUT_GPKG      = BASE / "crowns_labeled.gpkg"
MIXED_CSV     = BASE / "mixed_segments.csv"

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main() -> None:
    print("Reading data …")
    crowns  = gpd.read_file(CROWNS_GPKG)
    samples = gpd.read_file(TRAIN_GPKG)

    # CRS sanity check ------------------------------------------------
    if crowns.crs != samples.crs:
        samples = samples.to_crs(crowns.crs)

    # run the labelling ----------------------------------------------
    print("Labelling crowns …")
    clean_crowns, mixed_ids = label_segments(crowns, samples)

    # write outputs ---------------------------------------------------
    print(f"✓ labelled crowns : {len(clean_crowns):,} → {OUT_GPKG.name}")
    clean_crowns.to_file(OUT_GPKG, driver="GPKG")

    print(f"✓ mixed segments  : {len(mixed_ids):,} → {MIXED_CSV.name}")
    pd.Series(mixed_ids, name="segment_id").to_csv(MIXED_CSV, index=False)


if __name__ == "__main__":
    main()