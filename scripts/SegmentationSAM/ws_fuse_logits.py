#!/usr/bin/env python3
"""
Fuse SAM-logit tiles + cost tiles → crown polygons via Watershed.
Pure-Python, works on Python 3.12 with scikit-image wheels.

Inputs
------
  sam_logits_r<row>_c<col>.tif   – N-band logits
  cost_r<row>_c<col>.tif         – 1-band cost (0..1)

Outputs
-------
  crowns_r<row>_c<col>.gpkg      – polygons, field `seed_id`
"""
from pathlib import Path
import sys, click, numpy as np, rasterio, geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
from skimage.segmentation import watershed
from skimage.filters import sobel

@click.command()
@click.option('--logits-dir', type=click.Path(exists=True), required=True)
@click.option('--cost-dir',   type=click.Path(exists=True), required=True)
@click.option('--seeds',      type=click.Path(exists=True), required=True,
              help='canonical_seeds.gpkg with seed points')
@click.option('--out-dir',    type=click.Path(), required=True)
def main(logits_dir, cost_dir, seeds, out_dir):
    import geopandas as gpd
    seeds_gdf = gpd.read_file(seeds)

    logits_dir, cost_dir, out_dir = map(Path, (logits_dir, cost_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    for logp in sorted(logits_dir.glob('sam_logits_*.tif')):
        costp = cost_dir / logp.name.replace('sam_logits', 'cost')
        if not costp.exists():
            click.echo(f'⚠ cost tile missing for {logp.name}', err=True)
            continue

        with rasterio.open(logp) as L, rasterio.open(costp) as C:
            cost = C.read(1).astype(np.float32)          # (H,W)
            H, W = cost.shape

            # --- build marker grid ---------------------------------
            markers = np.zeros((H, W), np.int32)
            window_bounds = C.bounds
            seeds_tile = seeds_gdf[seeds_gdf.within(
                box(*window_bounds))].copy()
            if seeds_tile.empty:
                continue
            for idx, pt in enumerate(seeds_tile.geometry, start=1):
                col, row = C.index(pt.x, pt.y)
                markers[row, col] = idx

            # --- watershed on gradient of cost ---------------------
            elevation = sobel(cost)          # high at edges
            labels = watershed(elevation, markers, mask=(cost < 1))

            # --- vectorise polygons --------------------------------
            polys, ids = [], []
            for geom, val in shapes(labels.astype(np.int32),
                                    mask=labels > 0,
                                    transform=C.transform):
                polys.append(shape(geom))
                ids.append(int(val))
            gdf = gpd.GeoDataFrame({'seed_id': ids}, geometry=polys,
                                   crs=C.crs)

            out = out_dir / logp.with_suffix('.gpkg').name.replace('sam_logits', 'crowns')
            gdf.to_file(out, driver='GPKG', overwrite=True)
            click.echo(f'✓ {out.name}  (crowns: {len(gdf)})')

if __name__ == '__main__':
    try:
        from shapely.geometry import box  # cheap import inside main
        main()
    except Exception as e:
        click.echo(f'Error: {e}', err=True); sys.exit(1)