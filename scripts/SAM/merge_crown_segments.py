#!/usr/bin/env python3
"""
merge_crown_segments.py  –  crown-level *overlap* merge

1. Read every *.gpkg* already written by crown_segmentation.py
2. Cluster crowns whose Intersection-over-Union (IoU) ≥ IOU_MIN
3. Dissolve geometry + aggregate attributes inside each cluster
4. Save one master GeoPackage

The script is safe to re-run: the output file is overwritten each time.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
import shapely
import re
import os
from tqdm import tqdm

# ───── CONFIGURE ─────────────────────────────────────────────────────────
SRC_DIR   = Path("/Users/iosefa/repos/obia/scripts/SAM/crown_segments")
OUT_GPKG  = SRC_DIR.parent / "merged_crown_segments.gpkg"
LAYERNAME = "crown_segments"

IOU_MIN   = 0.55          # crowns with IoU ≥ this value will be merged
# ------------------------------------------------------------------------

sid_re = re.compile(r"\D*(\d+)\D*$")   # pull trailing number from filename

# 1 ── load every crown into one huge GDF ---------------------------------
gdfs, ref_crs = [], None
for gpkg in sorted(SRC_DIR.glob("*.gpkg")):
    gdf = gpd.read_file(gpkg)          # default layer
    m   = sid_re.match(gpkg.stem)
    gdf["sid"] = m.group(1) if m else gpkg.stem
    gdfs.append(gdf)

    if ref_crs is None:
        ref_crs = gdf.crs
    elif gdf.crs != ref_crs:
        raise ValueError(f"CRS mismatch in {gpkg}")

allcrowns = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=ref_crs)
allcrowns["__idx"] = allcrowns.index           # keep original row id
sindex = allcrowns.sindex

# 2 ── build IoU-based clusters ------------------------------------------
visited, clusters = set(), []                  # list of sets of row-ids

for idx, geom in enumerate(tqdm(allcrowns.geometry, desc="clustering")):
    if idx in visited:
        continue
    cluster = {idx}
    queue   = [idx]

    while queue:
        i = queue.pop()
        geom_i = allcrowns.at[i, "geometry"]

        # candidate neighbours = anything whose bbox intersects geom_i
        for j in sindex.query(geom_i, predicate="intersects"):
            if j in visited or j in cluster or j == i:
                continue
            geom_j = allcrowns.at[j, "geometry"]
            inter  = geom_i.intersection(geom_j).area
            union  = geom_i.union(geom_j).area
            if union and inter / union >= IOU_MIN:
                cluster.add(j)
                queue.append(j)

    visited.update(cluster)
    clusters.append(cluster)

# 3 ── dissolve geometries + aggregate attributes ------------------------
merged_rows = []
for cid, members in enumerate(tqdm(clusters, desc="dissolving")):
    subset = allcrowns.loc[list(members)]

    # dissolve geometry
    merged_geom = unary_union(subset.geometry)

    # build output record
    out = subset.iloc[0].drop(["geometry", "__idx"])    # copy attrs
    out["src_sid"] = ",".join(sorted(subset["sid"].unique()))
    out["n_merge"] = len(subset)
    merged_rows.append({**out.to_dict(), "geometry": merged_geom})

merged = gpd.GeoDataFrame(merged_rows, crs=ref_crs)

# 4 ── write ----------------------------------------------------------------
if OUT_GPKG.exists():
    os.remove(OUT_GPKG)

merged.to_file(OUT_GPKG, layer=LAYERNAME, driver="GPKG")
print(f"✓  wrote {len(merged)} merged crowns to {OUT_GPKG} (layer '{LAYERNAME}')")


# #!/usr/bin/env python3
# """
# merge_crown_segments.py
#
# Combine every .gpkg in `crown_segments/` into one master GeoPackage
# *without modifying* any geometry or existing attributes.
# """
#
# from pathlib import Path
# import re
# import geopandas as gpd
# import pandas as pd
# import os
#
# # ───── CONFIGURE ──────────────────────────────────────────────────────
# SRC_DIR   = Path("/Users/iosefa/repos/obia/scripts/SAM/crown_segments")
# OUT_GPKG  = SRC_DIR.parent / "all_crown_segments.gpkg"
# LAYERNAME = "crown_segments"
# # ---------------------------------------------------------------------
#
# sid_re = re.compile(r"\D*(\d+)\D*$")          # extract trailing number
# gdfs   = []                                   # will hold one GDF per file
# ref_crs = None
#
# for gpkg_path in sorted(SRC_DIR.glob("*.gpkg")):
#     # read the first (default) layer unchanged
#     gdf = gpd.read_file(gpkg_path)
#
#     # keep existing columns, just add a unique `sid`
#     fname = gpkg_path.stem
#     m = sid_re.match(fname)
#     gdf["sid"] = m.group(1) if m else fname
#
#     gdfs.append(gdf)
#
#     # simple CRS sanity‑check
#     if ref_crs is None:
#         ref_crs = gdf.crs
#     elif gdf.crs != ref_crs:
#         raise ValueError(f"CRS mismatch between {gpkg_path} and earlier files.")
#
# # concatenate rows → one big GeoDataFrame
# merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=ref_crs)
#
# # overwrite target GeoPackage every time
# if OUT_GPKG.exists():
#     os.remove(OUT_GPKG)
#
# merged.to_file(OUT_GPKG, layer=LAYERNAME, driver="GPKG")
# print(f"✓  wrote {len(merged)} crowns to {OUT_GPKG}  (layer '{LAYERNAME}')")