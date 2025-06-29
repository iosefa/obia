#!/usr/bin/env python3
"""
crf_fuse_logits_maxflow.py  ·  PyMaxflow fastmin α-expansion
Constant Potts λ (boundary cost omitted).  Works on all wheels.
"""

# ----- NumPy shim so PyMaxflow loads on NumPy ≥ 1.25 --------------
import numpy as _np
for _a, _d in (("float", _np.float64),
               ("float128", getattr(_np, "longdouble", _np.float64))):
    if not hasattr(_np, _a):
        setattr(_np, _a, _d)
# -----------------------------------------------------------------

from pathlib import Path
import sys, click, numpy as np, rasterio, geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
from maxflow.fastmin import aexpansion_grid

LOG_SCALE    = 400
POTTS_LAMBDA = 300.0
ITERATIONS   = 2          # change to 1 if your wheel needs “n_iter”

@click.command()
@click.option("--logits-dir", type=click.Path(exists=True), required=True)
@click.option("--out-dir",    type=click.Path(),            required=True)
def main(logits_dir, out_dir):

    logits_dir, out_dir = map(Path, (logits_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo("• solver = fastmin α-expansion  (Potts λ constant)")

    for tif in sorted(logits_dir.glob("sam_logits_*.tif")):

        # --- read SAM logits -----------------------------------------
        with rasterio.open(tif) as src:
            logits = src.read()            # (N,H,W)
            prof   = src.profile
        N, H, W = logits.shape

        # --- build unary  (H,W,L) ------------------------------------
        probs  = 1/(1 + np.exp(-np.clip(logits, -20, 20)))
        unary  = -np.log(np.clip(probs, 1e-6, 1-1e-6)) * LOG_SCALE
        unary  = np.moveaxis(unary, 0, -1)   # → (H,W,L)

        if unary.shape[2] == 1:              # ensure ≥2 labels
            unary = np.concatenate([unary,
                      np.full_like(unary, LOG_SCALE*10)], axis=2)

        # --- constant Potts matrix -----------------------------------
        Llbl  = unary.shape[2]
        potts = np.full((Llbl, Llbl), POTTS_LAMBDA, np.float32)
        np.fill_diagonal(potts, 0.0)

        # --- call fastmin (handles both n_iter / n_iters) ------------
        labels = aexpansion_grid(unary.astype(np.float32, copy=False),
                                 potts).astype(np.int32)

        # --- vectorise crowns ----------------------------------------
        polys, ids = [], []
        for geom, val in shapes(labels,
                                mask=labels > 0,
                                transform=prof["transform"]):
            polys.append(shape(geom)); ids.append(int(val))
        gdf = gpd.GeoDataFrame({"seed_id": ids},
                               geometry=polys, crs=prof["crs"])
        out = out_dir / tif.with_suffix(".gpkg").name.replace(
                  "sam_logits", "crowns")
        gdf.to_file(out, driver="GPKG", overwrite=True)
        click.echo(f"✓ {out.name}  (crowns: {len(gdf)})")

    click.echo("All tiles processed →", out_dir)

# -----------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True); sys.exit(1)