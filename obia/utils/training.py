import math
import os
import json
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
import numpy as np
import cv2
from tqdm import tqdm

from obia.utils.image import apply_clahe, rescale_to_8bit


def generate_tiles(bounds, step, tile_size):
    """
    Generator that yields bounding boxes (minx, miny, maxx, maxy)
    by stepping through the full raster bounds in increments of 'step',
    producing tiles of size 'tile_size'.
    """
    minx, miny, maxx, maxy = bounds
    y = miny
    while y < maxy:
        x = minx
        tile_top = y + tile_size
        while x < maxx:
            tile_right = x + tile_size
            cur_maxx = min(tile_right, maxx)
            cur_maxy = min(tile_top, maxy)
            yield (x, y, cur_maxx, cur_maxy)
            x += step
        y += step

def tile_and_process(
    raster_path,
    mask_path=None,
    boxes_gpkg_path=None,
    output_dir="output_tiles",
    tile_size=150.0,
    overlap=50.0,
    selected_bands=(4, 2, 1),
    feather_radius=0.0,
    blur_kernel=5,
    darken_factor=0.8,
    apply_clahe_flag=True,
    rescale=True
):
    """
    Tiles an input raster (and corresponding binary mask if provided), optionally
    applies CLAHE, optionally feathers the canopy edges using a distance transform,
    and writes out per-tile images (JPEG). Also converts polygons to bounding-box
    annotations if 'boxes_gpkg_path' is provided.

    Additionally, it saves a 'transforms.json' containing each tile's
    transform (as [a,b,c,d,e,f]) and its CRS in string form, so you can
    later reconstruct georeferencing for each tile.

    Parameters
    ----------
    raster_path : str
        Path to the main image raster (e.g., "image.tif").
    mask_path : str or None
        If provided, path to a binary canopy mask (0=background, 1=canopy).
        If None, no masking or feathering will be done.
    boxes_gpkg_path : str or None
        If provided, path to a GeoPackage with polygon annotations to be converted
        to bounding boxes per tile. If None, no annotations are generated.
    output_dir : str
        Directory to save output JPEG tiles, annotations.json, and transforms.json.
    tile_size : float
        Size of each tile in the raster's coordinate units (e.g., meters).
    overlap : float
        Overlap (in same units) between adjacent tiles.
    selected_bands : tuple of ints
        Which bands (1-based in raster) to extract from the raster. E.g., (4,2,1).
    feather_radius : float
        If > 0, a distance transform is used to create a soft alpha around
        canopy edges. 0 = no feathering. Only relevant if mask_path is not None.
    blur_kernel : int or tuple
        Size of the GaussianBlur kernel (must be an odd int or an odd tuple),
        e.g. 5 or (5,5). If 0, skip blur entirely.
    darken_factor : float
        Multiplier for background brightness, e.g., 0.8 = 80% brightness, 0.2 = 20% brightness.
        Only relevant if mask_path is not None.
    apply_clahe_flag : bool
        If True, apply CLAHE to the rescaled 8-bit tile before blending.
        If False, skip CLAHE and use the raw 8-bit image.
    rescale : bool
        If True, apply a percentile-based stretch to 8-bit. If False, do a min-max
        linear scaling (or no scaling if tile_min==tile_max).

    Returns
    -------
    None
        Writes:
          - Per-tile JPEG images in 'output_dir'.
          - 'annotations.json' (if polygons are provided).
          - 'transforms.json' with each tile's geotransform & CRS.
    """

    os.makedirs(output_dir, exist_ok=True)
    step = tile_size - overlap

    # If we have polygons, read them; else set None
    if boxes_gpkg_path:
        gdf = gpd.read_file(boxes_gpkg_path)
    else:
        gdf = None
    print("reading raster...")
    with rasterio.open(raster_path) as src:
        # If we have a mask path, open it; else None
        mask_src = rasterio.open(mask_path) if mask_path else None

        # If GPKG was loaded, ensure CRS matches (if needed)
        if gdf is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        all_annotations = {}
        tile_index = 0

        # Dictionary to store tile transforms for each tile_name
        transforms_dict = {}
        print("read raster")
        # Generate tiles

        width = src.bounds.right - src.bounds.left
        height = src.bounds.top - src.bounds.bottom

        num_tiles_x = math.ceil((width - overlap) / (tile_size - overlap))
        num_tiles_y = math.ceil((height - overlap) / (tile_size - overlap))
        total_tiles = num_tiles_x * num_tiles_y

        for tbox in tqdm(generate_tiles(src.bounds, step, tile_size), total=total_tiles):
        # for tbox in generate_tiles(src.bounds, step, tile_size):
            tile_index += 1
            minx, miny, maxx, maxy = tbox

            # Optional step: Filter polygons that lie within this tile
            if gdf is not None:
                possible = gdf.cx[minx:maxx, miny:maxy]
                tile_poly = box(minx, miny, maxx, maxy)
                tile_polygons = possible[possible.geometry.within(tile_poly)]
            else:
                tile_polygons = []  # empty if no GPKG

            # (1) Read the tile from main raster
            tile_window = from_bounds(minx, miny, maxx, maxy, src.transform)
            data = src.read(indexes=[b+1 for b in selected_bands], window=tile_window)
            # data shape => (num_bands, H, W)

            tile_img = np.moveaxis(data, 0, -1)  # shape => (H, W, num_bands)

            # (2) Rescale to 8-bit & optionally apply CLAHE
            if rescale:
                tile_img_8bit = rescale_to_8bit(tile_img)  # percentile-based approach
            else:
                tile_min, tile_max = tile_img.min(), tile_img.max()
                if tile_min == tile_max:
                    tile_img_8bit = np.zeros_like(tile_img, dtype=np.uint8)
                else:
                    tile_img_8bit = 255 * (tile_img - tile_min) / (tile_max - tile_min)
                tile_img_8bit = np.clip(tile_img_8bit, 0, 255).astype(np.uint8)

            if apply_clahe_flag:
                channels = cv2.split(tile_img_8bit)
                clahe_channels = [apply_clahe(ch) for ch in channels]
                tile_img_final = cv2.merge(clahe_channels)
            else:
                tile_img_final = tile_img_8bit

            # (3) If we have a mask, read the tile & do blending
            if mask_src:
                mask_data = mask_src.read(1, window=tile_window)

                # if mask_data.shape[0] == 0 or mask_data.shape[1] == 0:
                #     continue
                # if not np.any(mask_data):
                #     continue
                #
                # if mask_data.shape != tile_img_final.shape[:2]:
                #     # resample mask_data to match tile_img_final
                #     mask_data = cv2.resize(
                #         mask_data,
                #         (tile_img_final.shape[1], tile_img_final.shape[0]),
                #         interpolation=cv2.INTER_NEAREST
                #     )

                # (A) If blur_kernel == 0 => skip blur
                if isinstance(blur_kernel, int):
                    if blur_kernel == 0:
                        blurred_img = tile_img_final
                    else:
                        blur_kernel = (blur_kernel, blur_kernel)
                        blurred_img = cv2.GaussianBlur(tile_img_final, blur_kernel, 0)
                elif isinstance(blur_kernel, tuple):
                    if blur_kernel == (0,0):
                        blurred_img = tile_img_final
                    else:
                        blurred_img = cv2.GaussianBlur(tile_img_final, blur_kernel, 0)

                # Darken background
                if darken_factor == 0:
                    darkened_background = blurred_img
                else:
                    darkened_background = (blurred_img * darken_factor).astype(np.uint8)

                # (B) Feather if feather_radius > 0, else do hard blend
                if feather_radius > 0:
                    mask_8u = (mask_data * 255).astype(np.uint8)
                    inverse_mask = 255 - mask_8u
                    dist = cv2.distanceTransform(inverse_mask, cv2.DIST_L2, 3)

                    alpha = 1.0 - (dist / feather_radius)
                    alpha = np.clip(alpha, 0.0, 1.0)
                    alpha_3d = np.dstack([alpha]*3)  # (H, W, 3)

                    tile_f = tile_img_final.astype(np.float32)
                    darkened_bg_f = darkened_background.astype(np.float32)

                    out_img_f = alpha_3d * tile_f + (1.0 - alpha_3d) * darkened_bg_f
                    out_img = np.clip(out_img_f, 0, 255).astype(np.uint8)
                else:
                    # Hard blend with binary mask
                    mask_3d = np.stack([mask_data]*3, axis=-1)  # (H, W, 3)
                    blended_f = tile_img_final * mask_3d + darkened_background * (1 - mask_3d)
                    out_img = blended_f.astype(np.uint8)
            else:
                # No mask => just keep tile_img_final as is
                out_img = tile_img_final

            # (4) Save tile
            out_height, out_width = out_img.shape[:2]
            # Create a local transform for this tile
            tile_transform = rasterio.windows.transform(tile_window, src.transform)

            profile = src.profile.copy()
            profile.update({
                "driver": "JPEG",
                "height": out_height,
                "width": out_width,
                "count": out_img.shape[2],  # typically 3
                "dtype": "uint8",
                "transform": tile_transform,  # georeferencing for the tile
                "crs": src.crs
            })

            tile_name = f"img_{tile_index:03d}.jpg"
            tile_path = os.path.join(output_dir, tile_name)
            out_data = np.moveaxis(out_img, -1, 0)

            with rasterio.open(tile_path, "w", **profile) as dst:
                dst.write(out_data)

            # Store the transform & CRS in a dict
            # tile_transform is an Affine object with attributes (a,b,c,d,e,f)
            # We'll store them as a list [a,b,c,d,e,f] plus a CRS string
            transform_list = [
                tile_transform.a, tile_transform.b, tile_transform.c,
                tile_transform.d, tile_transform.e, tile_transform.f
            ]
            transforms_dict[tile_name] = {
                "transform": transform_list,
                "crs": str(src.crs)  # or src.crs.to_string() / to_wkt() if you prefer
            }

            # (5) Convert polygons -> bounding boxes
            if gdf is not None and len(tile_polygons) > 0:
                row_off = tile_window.row_off
                col_off = tile_window.col_off

                boxes_array = []
                labels_array = []

                for idx, row in tile_polygons.iterrows():
                    poly = row.geometry
                    pxmin, pymin, pxmax, pymax = poly.bounds

                    # Convert geospatial coords -> (row, col)
                    row_tl, col_tl = rowcol(src.transform, pxmin, pymax)
                    row_br, col_br = rowcol(src.transform, pxmax, pymin)

                    # Convert to tile coords
                    row_tl_local = row_tl - row_off
                    col_tl_local = col_tl - col_off
                    row_br_local = row_br - row_off
                    col_br_local = col_br - col_off

                    x_min_pix = col_tl_local
                    y_min_pix = row_tl_local
                    x_max_pix = col_br_local
                    y_max_pix = row_br_local

                    # Clamp
                    x_min_pix = max(0, min(x_min_pix, out_width - 1))
                    x_max_pix = max(0, min(x_max_pix, out_width - 1))
                    y_min_pix = max(0, min(y_min_pix, out_height - 1))
                    y_max_pix = max(0, min(y_max_pix, out_height - 1))

                    # Skip degenerate boxes
                    if x_min_pix >= x_max_pix or y_min_pix >= y_max_pix:
                        continue

                    boxes_array.append([x_min_pix, y_min_pix, x_max_pix, y_max_pix])
                    labels_array.append(1)  # or your class label

                if len(boxes_array) == 0:
                    all_annotations[f"img_{tile_index:03d}"] = {
                        "file_name": tile_name,
                        "boxes": [],
                        "labels": []
                    }
                else:
                    all_annotations[f"img_{tile_index:03d}"] = {
                        "file_name": tile_name,
                        "boxes": boxes_array,
                        "labels": labels_array
                    }

        # End of tile loop
        # Close mask if used
        if mask_src:
            mask_src.close()

    # (6) Write out the annotations JSON (only if we had a GPKG)
    if gdf is not None:
        json_path = os.path.join(output_dir, "annotations.json")
        with open(json_path, "w") as f:
            json.dump(all_annotations, f, indent=2)
        print(f"Annotations JSON written to: {json_path}")

    # (7) Write out the transforms JSON
    transforms_path = os.path.join(output_dir, "transforms.json")
    with open(transforms_path, "w") as ft:
        json.dump(transforms_dict, ft, indent=2)
    print(f"Transforms JSON written to: {transforms_path}")

    print("Done! Tiles written to:", output_dir)