import json
import geopandas as gpd
import numpy as np

from affine import Affine
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import Polygon
from typing import List, Tuple


def label_segments(segments: gpd.GeoDataFrame, labelled_points: gpd.GeoDataFrame) -> Tuple[
    gpd.GeoDataFrame, List[str]]:
    """
    :param segments: A GeoDataFrame representing the segments to be labeled.
    :param labelled_points: A GeoDataFrame representing the labeled points used for segment labeling.
    :return: A tuple containing a GeoDataFrame with labeled segments and a list of segment IDs for mixed segments.
    """
    mixed_segments = []
    labelled_segments = segments.copy()
    intersections = gpd.sjoin(labelled_segments, labelled_points, how='inner', predicate='intersects')

    for polygon_id, group in intersections.groupby(intersections.index):
        classes = group['class'].unique()

        if len(classes) == 1:
            labelled_segments.loc[polygon_id, 'feature_class'] = classes[0]
        else:
            segment_id = group['segment_id'].values[0]
            mixed_segments.append(segment_id)

    labelled_segments = labelled_segments[labelled_segments['feature_class'].notna()]

    return labelled_segments, mixed_segments


def crop_image_to_bbox(image, geom):
    """
    Crop the image data to the bounding box of the given geometry.

    :param image: The Image object containing the image data and rasterio object.
    :param geom: The geometry (Polygon) used to derive the bounding box for cropping.
    :return: Cropped image data as a NumPy array and the updated transform.
    """
    xmin, ymin, xmax, ymax = geom.bounds
    window = from_bounds(xmin, ymin, xmax, ymax, transform=image.transform)
    cropped_img_data = image.rasterio_obj.read(window=window)
    cropped_transform = image.rasterio_obj.window_transform(window)

    return cropped_img_data, cropped_transform


def mask_image_with_polygon(cropped_img_data, polygon, cropped_transform):
    """
    Masks all pixels outside the polygon for the given cropped image.

    :param cropped_img_data: The cropped image data as a NumPy array.
    :param polygon: The geometry (Polygon) used for masking.
    :param cropped_transform: The affine transform for the cropped image.
    :return: Masked image data as a NumPy array.
    """
    height, width = cropped_img_data.shape[1], cropped_img_data.shape[2]
    mask = geometry_mask([polygon], transform=cropped_transform, invert=True, out_shape=(height, width))
    mask_expanded = np.expand_dims(mask, axis=0)  # Add an extra dimension for bands
    masked_img_data = np.where(mask_expanded, cropped_img_data, np.nan)

    return masked_img_data


def save_deepforest_predictions_to_gpkg(
    df,               # DataFrame from DeepForest predict_image(...)
    tile_name,        # e.g. "img_007.jpg"
    transforms_json,  # path to transforms.json
    output_gpkg       # path to output .gpkg
):
    """
    Convert DeepForest predictions (pixel coords) to georeferenced polygons
    and save them in a GeoPackage.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ["xmin","ymin","xmax","ymax","label","score"].
        Optional "image_path" if you prefer to verify or filter by tile_name.
    tile_name : str
        The tile's filename (e.g. "img_007.jpg") that matches the key in transforms.json.
    transforms_json : str
        File path to the transforms.json created by tile_and_process(...).
        We'll load the tile's Affine transform + CRS from there.
    output_gpkg : str
        Path to the output GeoPackage file to create/overwrite.
    """
    # 1) Load transforms.json
    with open(transforms_json, "r") as f:
        transforms_dict = json.load(f)

    if tile_name not in transforms_dict:
        # todo: raise warning or something
        print(f"Tile '{tile_name}' not found in transforms.json. Skipping.")
        return

    # 2) Reconstruct the tile's Affine + CRS
    # transforms_dict[tile_name] = { "transform": [a,b,c,d,e,f], "crs": "EPSG:..." }
    tinfo = transforms_dict[tile_name]
    a, b, c, d, e, f = tinfo["transform"]
    tile_affine = Affine(a, b, c, d, e, f)
    crs_str = tinfo["crs"]  # e.g. "EPSG:32610"

    # 3) Convert each bounding box to a polygon in map coords
    records = []
    for idx, row in df.iterrows():
        # read pixel coords
        xmin_pix = row["xmin"]
        ymin_pix = row["ymin"]
        xmax_pix = row["xmax"]
        ymax_pix = row["ymax"]

        # top-left corner => (col=xmin_pix, row=ymin_pix)
        x1, y1 = tile_affine * (xmin_pix, ymin_pix)
        # top-right corner => (col=xmax_pix, row=ymin_pix)
        x2, y2 = tile_affine * (xmax_pix, ymin_pix)
        # bottom-right corner => (col=xmax_pix, row=ymax_pix)
        x3, y3 = tile_affine * (xmax_pix, ymax_pix)
        # bottom-left corner => (col=xmin_pix, row=ymax_pix)
        x4, y4 = tile_affine * (xmin_pix, ymax_pix)

        # build polygon (4-corner bounding box)
        geom = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)])

        # store relevant columns
        rec = {
            "label": row.get("label", "Tree"),   # default label if not present
            "score": row.get("score", None),
            "geometry": geom
        }
        records.append(rec)

    if not records:
        # todo: raise warning or something
        print(f"No predictions to save for tile {tile_name}")
        return

    gdf = gpd.GeoDataFrame(records, crs=crs_str)
    # 4) Save to GPKG
    gdf.to_file(output_gpkg, driver="GPKG")
