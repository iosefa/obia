import geopandas as gpd, pandas as pd
from pathlib import Path

SEGMENTS_GPKG = Path("../docs/example_data/slic_segments_full.tif/segments.gpkg")
LABELS_PARQUET = Path("segments_labeled.parquet")   # or .csv if you switched
CROWNS_OUT  = Path("crowns_panoptic_2.gpkg")
LAYER_NAME  = "crowns"

segs = gpd.read_file(SEGMENTS_GPKG)

# ensure seg_id is a normal column
if 'seg_id' not in segs.columns:
    segs = segs.reset_index().rename(columns={'index': 'seg_id'})

labels = pd.read_parquet(LABELS_PARQUET)   # use read_csv() if csv

segs = segs.merge(labels, on="seg_id", how="inner")

crowns = segs.dissolve(by="tree_id").reset_index()
crowns["area_m2"] = crowns.area
crowns = crowns[crowns.area_m2 > 4]        # drop tiny polygons

crowns.to_file(CROWNS_OUT, layer=LAYER_NAME, driver="GPKG")
print("✓ crowns  →", CROWNS_OUT, "with", len(crowns), "trees")