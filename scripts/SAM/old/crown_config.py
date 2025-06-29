# crown_config.py
from pathlib import Path

BASE_DIR        = Path("/Users/iosefa/repos/obia/scripts/SAM")
PROB_DIR        = BASE_DIR / "tmp_prob"
SLIC_FILE       = BASE_DIR / "slic.gpkg"
SEED_FILE       = BASE_DIR / "seeds_cost.gpkg"
COST_FILE       = BASE_DIR / "cost_r0_c1536.tif"
OUT_DIR         = BASE_DIR / "crown_segments"
CHM_FILE          = BASE_DIR / "chm.tif"

# weights that trade off the three layers:  w_prob + w_chm + w_cost = 1
W_PROB, W_CHM, W_COST = 0.5, 0.3, 0.2

GAUSSIAN_SIGMA  = 30       # pixels
PROB_THRESHOLD  = 0.15    # keep polygons whose mean P > this
EDGE_MIN_FRAC   = 0.2    # drop a neighbor if shared edge < 5 % of its perimeter
COST_MAX        = 0.5    # e.g. 30; set None to ignore cost raster

# geometric constraints
MAX_RADIUS        = 20          # metres
COMPACTNESS_MAX   = 5        # 1 = circle; higher allows less round shapes