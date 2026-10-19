"""Recreate the bundled Chesapeake crops using HTTP range reads."""

import argparse
import hashlib
import json
from pathlib import Path

import rasterio
from rasterio.windows import Window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parents[1] / "obia/datasets/chesapeake"
    if args.output_dir.resolve() == source_dir:
        parser.error("Use a separate output directory, then review regenerated files.")
    manifest = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT=45):
        for sample in manifest["samples"].values():
            window = Window(**sample["window"])
            for kind, entry in sample["files"].items():
                with rasterio.open(entry["url"]) as src:
                    pixels = src.read(window=window)
                    transform = src.window_transform(window)
                    if (hashlib.sha256(pixels.tobytes()).hexdigest() != entry["pixel_sha256"]
                            or src.crs.to_string() != sample["crs"]
                            or list(transform)[:6] != sample["transform"]):
                        raise ValueError(f"Source pixels or grid changed: {entry['url']}")
                    destination = args.output_dir / entry["filename"]
                    with rasterio.open(
                        destination, "w", driver="GTiff", height=int(window.height),
                        width=int(window.width), count=src.count, dtype=src.dtypes[0],
                        crs=src.crs, transform=transform, compress="deflate", predictor=2,
                        tiled=True, blockxsize=256, blockysize=256,
                        nodata=15 if kind == "reference" else src.nodata,
                    ) as dst:
                        dst.write(pixels)
                        dst.update_tags(source_url=entry["url"], upstream_split=sample["upstream_split"])
                    print(destination)


if __name__ == "__main__":
    main()
