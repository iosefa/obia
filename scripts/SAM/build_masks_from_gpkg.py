#!/usr/bin/env python3
# build_masks_per_cluster.py  (regex‑driven version)

from pathlib import Path
import os, re
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.ops import unary_union
from shapely import speedups
from tqdm import tqdm

if speedups.available:
    speedups.enable()

# ─── paths ───────────────────────────────────────────────────────────
BASE_DIR   = Path("/Users/iosefa/repos/obia/scripts/SAM")
TILE_DIR   = BASE_DIR / "composite_tiles"              # seed_<id>_comp.tif
MASK_DIR   = Path("data/train/masks")
SEED_GPKG  = BASE_DIR / "seeds_cost.gpkg"
CROWN_GPKG = "/Users/iosefa/repos/obia/docs/example_data/site_2/training_crowns1.gpkg"
# --------------------------------------------------------------------

MASK_DIR.mkdir(parents=True, exist_ok=True)

# 1 ── build {cluster_id: tile_path} from actual filenames
tile_regex = re.compile(r"seed_(\d+)_comp\.tif$", re.IGNORECASE)
tile_map   = {}
for p in TILE_DIR.glob("seed_*_comp.tif"):
    m = tile_regex.search(p.name)
    if m:
        tile_map[int(m.group(1))] = p

# 2 ── load vectors
seeds  = gpd.read_file(SEED_GPKG)
crowns = gpd.read_file(CROWN_GPKG).to_crs(seeds.crs)

skipped = {"none": [], "multi": [], "no_tile": []}
written = 0

for cid in tqdm(sorted(seeds["cluster"].unique()), desc="clusters"):
    tile_path = tile_map.get(int(cid))
    if tile_path is None:
        skipped["no_tile"].append(cid)
        continue

    # union of all seed geometries in this cluster
    seed_geom = unary_union(seeds.loc[seeds["cluster"] == cid, "geometry"])

    # crowns intersecting that seed area
    hit = crowns[crowns.intersects(seed_geom)]

    if len(hit) == 0:
        skipped["none"].append(cid)
        continue
    if len(hit) > 1:
        skipped["multi"].append(cid)
        continue

    crown_geom = hit.geometry.values[0]

    # rasterise crown to 0/1 mask aligned with the tile
    with rasterio.open(tile_path) as src:
        mask = rasterize(
            [crown_geom],
            out_shape=src.shape,
            transform=src.transform,
            fill=0, all_touched=True, dtype="uint8"
        )
        mask_path = MASK_DIR / tile_path.name.replace("_comp.tif", "_mask.tif")
        meta = src.meta.copy()
        meta.update(count=1, dtype="uint8", nodata=0)
        if mask_path.exists():
            os.remove(mask_path)
        with rasterio.open(mask_path, "w", **meta) as dst:
            dst.write(mask, 1)

    written += 1

# 3 ── summary
print(f"\n✓ wrote {written} masks to {MASK_DIR}")

for k, lst in skipped.items():
    if lst:
        print(f"• skipped {len(lst)} clusters ({k})  →  {list(lst)[:10]} …")