#!/usr/bin/env python3
"""
slice_cost_for_logits.py  (bounds-driven)
Cut a global cost.tif so every SAM-logit tile gets a perfectly
matching cost tile, whatever its size or grid.

• Works with any filename pattern:  reads bounds from the COG itself
• If cost.tif pixel size differs, it automatically re-samples
  (bilinear) to the logit grid.
"""

import click, rasterio, re
from pathlib import Path

@click.command()
@click.option("--cost",      type=click.Path(exists=True), required=True)
@click.option("--logits-dir", type=click.Path(exists=True), required=True)
@click.option("--out-dir",   type=click.Path(), required=True)
def main(cost, logits_dir, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(cost) as cost_ds:
        for log_path in Path(logits_dir).glob("*.tif"):
            with rasterio.open(log_path) as lds:
                # read bounds & profile from logit tile
                bounds   = lds.bounds
                lprofile = lds.profile

            # window in cost.tif that covers those bounds
            win = rasterio.windows.from_bounds(
                    *bounds, transform=cost_ds.transform,
                    width=lprofile['width'], height=lprofile['height'])

            # read & (if needed) resample to match log tile grid
            cost_arr = cost_ds.read(
                1,
                window=win,
                out_shape=(lprofile['height'], lprofile['width']),
                resampling=rasterio.enums.Resampling.bilinear)

            out = out_dir / log_path.name.replace("sam_logits", "cost")
            lprofile.update(driver="GTiff", compress="deflate", count=1, dtype="float32")
            with rasterio.open(out, "w", **lprofile) as dst:
                dst.write(cost_arr.astype("float32"), 1)
            print("✓", out.name)

    print("All matching cost tiles written →", out_dir)

if __name__ == "__main__":
    main()