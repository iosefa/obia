import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from obia.pointcloud import add_pointcloud_features
from obia.pointcloud.features import calculate_pointcloud_features


def test_calculate_pointcloud_features_for_one_segment():
    points = pd.DataFrame(
        {
            "z": [1.0, 2.0, 5.0],
            "intensity": [10.0, 20.0, 40.0],
        }
    )

    features = calculate_pointcloud_features(points, area=2.0)

    assert features["pc_point_count"] == 3
    assert features["pc_density"] == 1.5
    assert features["pc_z_mean"] == np.mean([1.0, 2.0, 5.0])
    assert features["pc_z_p95"] == np.percentile([1.0, 2.0, 5.0], 95)
    assert features["pc_intensity_mean"] == np.mean([10.0, 20.0, 40.0])


def test_add_pointcloud_features_preserves_segment_rows():
    segments = gpd.GeoDataFrame(
        {"segment_id": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:32605",
    )
    points = pd.DataFrame(
        {
            "x": [0.25, 0.75, 1.25],
            "y": [0.25, 0.75, 0.25],
            "z": [2.0, 4.0, 10.0],
            "intensity": [5.0, 7.0, 9.0],
        }
    )

    result = add_pointcloud_features(segments, points)

    assert list(result["segment_id"]) == [1, 2]
    assert list(result["pc_point_count"]) == [2, 1]
    assert result.loc[0, "pc_z_mean"] == 3.0
    assert result.loc[1, "pc_z_mean"] == 10.0
    assert result.loc[0, "pc_density"] == 2.0
