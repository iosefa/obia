import os
import math

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

from osgeo import gdal
from rasterio.features import rasterize
from shapely import Polygon
from shapely.geometry import box
from affine import Affine

from obia.handlers.geotif import Image
from obia.segmentation.segment_boundaries import create_segments


def get_raster_bbox(dataset):
    """
    :param dataset: Input dataset from which to retrieve the bounding box. Must have GetGeoTransform, RasterXSize, and RasterYSize methods.
    :return: A tuple representing the bounding box (min_x, min_y, max_x, max_y).
    """
    transform = dataset.GetGeoTransform()

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    min_x = transform[0]
    max_y = transform[3]
    max_x = min_x + width * transform[1]
    min_y = max_y + height * transform[5]

    return (min_x, min_y, max_x, max_y)


def _create_tile(dataset, i_offset, j_offset, w, h, binary_mask=False):
    tile_transform = dataset.GetGeoTransform()
    tile_transform = Affine(
        tile_transform[1], tile_transform[2], tile_transform[0] + i_offset * tile_transform[1],
        tile_transform[4], tile_transform[5], tile_transform[3] + j_offset * tile_transform[5]
    )
    if binary_mask:
        mask_data = dataset.GetRasterBand(1).ReadAsArray(i_offset, j_offset, w, h).astype(bool)
        return mask_data
    else:
        img_data = np.empty((h, w, dataset.RasterCount), dtype=np.float32)
        for band in range(1, dataset.RasterCount + 1):
            data = dataset.GetRasterBand(band).ReadAsArray(i_offset, j_offset, w, h)
            if data is not None:
                img_data[:, :, band - 1] = data

        crs = dataset.GetProjection()
        if crs is None:
            crs = "epsg:32702"
        affine_transformation = [tile_transform.a, tile_transform.b, tile_transform.d, tile_transform.e,
                                 tile_transform.c, tile_transform.f]
        image = Image(img_data, crs, affine_transformation, tile_transform, None)
        return image


def create_tiled_segments(input_raster, output_dir, input_mask=None,
                          method="slic", tile_size=200, buffer=30, crown_radius=5,
                          **kwargs):
    """
    :param input_raster: Path to the input raster file.
    :param output_dir: Directory where output files will be saved.
    :param input_mask: Optional path to an input mask file for masking specific regions.
    :param method: Segmentation method to be used, defaults to "slic". Currently, only the 'slic' method is supported.
    :param tile_size: Size of the tiles into which the raster is divided, default is 200.
    :param buffer: Buffer size for tile overlap, default is 30.
    :param crown_radius: Radius used to compute the number of segments/crowns, default is 5.
    :param kwargs: Additional keyword arguments passed to the segmentation function.
    :return: None
    """
    if method != "slic":
        raise ValueError("Currently, only the 'slic' method is supported for segmentation.")
    # buffer = buffer * 2
    dataset = gdal.Open(input_raster)
    if not dataset:
        raise ValueError(f"Unable to open {input_raster} or {input_mask}")

    mask_dataset = None
    # todo: mask should be optional
    if input_mask is not None:
        mask_dataset = gdal.Open(input_mask)
        if not mask_dataset:
            raise ValueError(f"Unable to open {input_mask}")

    # todo: dont mask segments that intersect with outer bbox
    outer_bbox = get_raster_bbox(dataset)

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    os.makedirs(output_dir, exist_ok=True)

    tile_index = 0

    all_black_segments = gpd.GeoDataFrame(columns=["geometry"])
    all_white_segments = gpd.GeoDataFrame(columns=["geometry"])

    for j in range(0, height, tile_size):
        for i in range(0, width, tile_size):
            is_white_tile = (i // tile_size + j // tile_size) % 2 != 0

            if is_white_tile:
                continue

            i_offset = i
            j_offset = j

            w = min(tile_size, width - i_offset)
            h = min(tile_size, height - j_offset)

            if w == 0 or h == 0:
                continue

            image = _create_tile(dataset, i_offset, j_offset, w, h)

            mask = None

            if mask_dataset:
                mask = _create_tile(mask_dataset, i_offset, j_offset, w, h, binary_mask=True)

            n_segments = kwargs.get("n_segments", None)
            if n_segments is None:
                geo_transform = dataset.GetGeoTransform()
                pixel_width = geo_transform[1]
                pixel_height = abs(geo_transform[5])
                pixel_area = pixel_width * pixel_height
                crown_area = math.pi * (crown_radius ** 2)
                tree_area = mask.sum() * pixel_area
                n_crowns = round(tree_area / crown_area)
                n_segments = n_crowns
            try:
                segmented_image = create_segments(
                    image=image,
                    mask=mask,
                    n_segments=n_segments,
                    method="slic",
                    **kwargs
                )

                # bbox = segmented_image.total_bounds
                # tile_boundary = box(bbox[0], bbox[1], bbox[2], bbox[3])
                # segmented_image_filtered = segmented_image[~segmented_image.intersects(tile_boundary.boundary)]
                all_black_segments = pd.concat([all_black_segments, segmented_image], ignore_index=True)
            except ValueError:
                print(f"empty tile: ({j}) ({i})")

            tile_index += 1
        tile_index += 1

    tile_index = 0
    for j in range(0, height, tile_size):
        for i in range(0, width, tile_size):
            is_white_tile = (i // tile_size + j // tile_size) % 2 != 0

            if is_white_tile:
                # ---- horizontal offsets & width ----
                i_offset = max(0, i - buffer)  # shift left by buffer unless we hit 0
                right_edge = min(width, i + tile_size + buffer)  # shift right by buffer unless we hit width
                w = right_edge - i_offset  # final window width

                # ---- vertical offsets & height ----
                j_offset = max(0, j - buffer)
                bottom_edge = min(height, j + tile_size + buffer)
                h = bottom_edge - j_offset

                w = max(0, min(w, width - i_offset))
                h = max(0, min(h, height - j_offset))
                if w == 0 or h == 0:
                    continue

                # create tile mask and image
                image = _create_tile(dataset, i_offset, j_offset, w, h)
                mask = None
                if mask_dataset:
                    mask = _create_tile(mask_dataset, i_offset, j_offset, w, h, binary_mask=True)

                tile_transform = image.transform
                left, top = tile_transform * (0, 0)
                right, bottom = tile_transform * (w, h)
                bbox = (left, bottom, right, top)

                tile_polygon = box(*bbox)

                corner_length = buffer / 2
                minx, miny, maxx, maxy = tile_polygon.bounds
                bottom_left_square = Polygon([
                    (minx, miny),
                    (minx + corner_length, miny),
                    (minx + corner_length, miny + corner_length),
                    (minx, miny + corner_length)
                ])
                bottom_right_square = Polygon([
                    (maxx - corner_length, miny),
                    (maxx, miny),
                    (maxx, miny + corner_length),
                    (maxx - corner_length, miny + corner_length)
                ])
                tile_polygon = tile_polygon.difference(bottom_left_square).difference(bottom_right_square)

                intersecting_black_segments = all_black_segments[
                    all_black_segments.within(tile_polygon) | all_black_segments.overlaps(tile_polygon)
                    ]
                intersecting_white_segments = all_white_segments[
                    all_white_segments.within(tile_polygon) | all_white_segments.overlaps(tile_polygon)
                    ]

                if not intersecting_black_segments.empty or not intersecting_white_segments.empty:
                    overlapping_black_segments_for_mask = intersecting_black_segments[
                        intersecting_black_segments.overlaps(tile_polygon)
                    ]
                    overlapping_white_segments_for_mask = intersecting_white_segments[
                        intersecting_white_segments.overlaps(tile_polygon)
                    ]

                    overlapping_black_segments_for_delete = intersecting_black_segments[
                        intersecting_black_segments.within(tile_polygon)
                    ]
                    overlapping_white_segments_for_delete = intersecting_white_segments[
                        intersecting_white_segments.within(tile_polygon)
                    ]

                    indices_to_delete_black = overlapping_black_segments_for_delete.index
                    all_black_segments = all_black_segments.drop(indices_to_delete_black)

                    indices_to_delete_white = overlapping_white_segments_for_delete.index
                    all_white_segments = all_white_segments.drop(indices_to_delete_white)

                    combined_geometries = []

                    if not overlapping_white_segments_for_mask.empty:
                        white_geometries = [
                            (segment.geometry, 1) for _, segment in overlapping_white_segments_for_mask.iterrows()
                        ]
                        combined_geometries.extend(white_geometries)
                    if not overlapping_black_segments_for_mask.empty:
                        black_geometries = [
                            (segment.geometry, 1) for _, segment in overlapping_black_segments_for_mask.iterrows()
                        ]
                        combined_geometries.extend(black_geometries)
                    corner_geometries = [(bottom_left_square, 1), (bottom_right_square, 1)]
                    combined_geometries.extend(corner_geometries)

                    mask_rasterized = rasterize(
                        combined_geometries,
                        out_shape=(image.img_data.shape[0], image.img_data.shape[1]),
                        transform=image.transform,
                        fill=0,
                        default_value=1,
                        dtype=rasterio.uint8
                    )

                    if mask is not None:
                        mask[mask_rasterized == 1] = 0
                    else:
                        mask = mask_rasterized
                else:
                    print(f"No overlapping black segments found for tile ({i}, {j}).")

                n_segments = kwargs.get("n_segments", None)
                if n_segments is None:
                    geo_transform = dataset.GetGeoTransform()
                    pixel_width = geo_transform[1]
                    pixel_height = abs(geo_transform[5])
                    pixel_area = pixel_width * pixel_height
                    crown_area = math.pi * (crown_radius ** 2)
                    tree_area = mask.sum() * pixel_area
                    n_crowns = round(tree_area / crown_area)
                    n_segments = n_crowns
                try:
                    segmented_image = create_segments(
                        image=image,
                        mask=mask,
                        n_segments=n_segments,
                        method="slic",
                        **kwargs
                    )
                    all_white_segments = pd.concat([all_white_segments, segmented_image], ignore_index=True)
                except ValueError:
                    print(f"empty tile: ({i}, {j}).")

            tile_index += 1
        tile_index += 1

    all_segments = pd.concat([all_black_segments, all_white_segments], ignore_index=True)
    all_segments['segment_id'] = range(1, len(all_segments) + 1)
    all_segments.to_file(os.path.join(output_dir, "segments.gpkg"), driver="GPKG")
