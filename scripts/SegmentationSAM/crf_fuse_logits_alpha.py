#!/usr/bin/env python3
"""
crf_fuse_logits_alpha.py  ·  gco-wrapper 3.0.9 safe

Uses only cut_grid_graph_simple (Potts) which every wheel provides.
Pair-wise λ is constant; boundary cost variation is not supported by
this API.
"""

# NumPy shim for NumPy ≥1.25
import numpy as _np
for n, d in (("float", _np.float64),
             ("float128", getattr(_np, "longdouble", _np.float64))):
    if not hasattr(_np, n):
        setattr(_np, n, d)

from pathlib import Path
import sys, click, numpy as np, rasterio, geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
import gco

_GRID = gco.cut_grid_graph_simple          # only function we can rely on
LOG_SCALE   = 400                          # −log(p) × this
SAFE_MAX    = 100.0                        # after rescale  → 100 × 1000 = 1e5
BASE_LAMBDA = 10.0                         # constant Potts weight
ITER        = 5

@click.command()
@click.option("--logits-dir", type=click.Path(exists=True), required=True)
@click.option("--out-dir",    type=click.Path(),            required=True)
def main(logits_dir, out_dir):

    logits_dir, out_dir = map(Path, (logits_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo("• solver = cut_grid_graph_simple  (Potts λ constant)")

    for tif in sorted(logits_dir.glob("sam_logits_*.tif")):

        with rasterio.open(tif) as src:
            logits = src.read()                       # (N,H,W)
            prof   = src.profile
        N, H, W = logits.shape

        # unary cube ----------------------------------------------------
        probs  = 1/(1 + np.exp(-np.clip(logits, -20, 20)))
        unary  = -np.log(np.clip(probs, 1e-6, 1-1e-6)) * LOG_SCALE
        unary  = np.moveaxis(unary, 0, -1)             # (H,W,L)

        if unary.shape[2] == 1:                        # ≥2 labels
            unary = np.concatenate([unary,
                      np.full_like(unary, LOG_SCALE*10)], axis=2)

        # rescale so max ≤ 100
        s       = SAFE_MAX / unary.max()
        unary_f = np.ascontiguousarray(unary * s, dtype=np.float64)

        lam_f   = BASE_LAMBDA * s
        Llbl    = unary_f.shape[2]
        potts   = np.full((Llbl, Llbl), lam_f, np.float64)
        np.fill_diagonal(potts, 0.0)
        potts   = np.ascontiguousarray(potts)

        # α-expansion ---------------------------------------------------
        labels  = _GRID(unary_f, potts,
                        n_iter=ITER, algorithm="expansion")

        # vectorise crowns ---------------------------------------------
        polys, ids = [], []
        for geom, val in shapes(labels.astype(np.int32),
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

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True); sys.exit(1)