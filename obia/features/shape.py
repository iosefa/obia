"""Polygon shape measurements with explicit coordinate units."""

import numpy as np
from shapely.affinity import affine_transform


SHAPE_FEATURES = (
    "area", "perimeter", "major_axis_length", "minor_axis_length", "elongation",
    "circularity", "solidity", "rectangularity", "orientation", "hole_count",
)


def shape_features(geometry, features, transform=None):
    if transform is not None:
        inverse = ~transform
        geometry = affine_transform(geometry, [inverse.a, inverse.b, inverse.d,
                                               inverse.e, inverse.c, inverse.f])
    result = {name: np.nan for name in features}
    if geometry.is_empty:
        return result
    area, perimeter = geometry.area, geometry.length
    hull = geometry.convex_hull
    # Supported Shapely versions can emit these warnings for valid axis-aligned
    # rectangles; verify the returned geometry before using its dimensions.
    with np.errstate(divide="ignore", invalid="ignore"):
        rectangle = geometry.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon" or rectangle.area == 0:
        return result
    coords = np.asarray(rectangle.exterior.coords)
    vectors = np.diff(coords, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    major, minor = lengths.max(), lengths.min()
    # Resolve square/tied-axis orientation deterministically to the smaller angle.
    directions = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 180
    direction = directions[np.isclose(lengths, major, rtol=1e-10, atol=0)].min()
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    values = {
        "area": area, "perimeter": perimeter,
        "major_axis_length": major, "minor_axis_length": minor,
        "elongation": major / minor,
        "circularity": 4 * np.pi * area / perimeter ** 2,
        "solidity": area / hull.area,
        "rectangularity": area / rectangle.area,
        "orientation": direction,
        "hole_count": sum(len(polygon.interiors) for polygon in polygons),
    }
    return {name: values[name] for name in features}
