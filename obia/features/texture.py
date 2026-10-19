"""Texture measurements that exclude invalid and out-of-object pixel pairs."""

from dataclasses import dataclass
import math

import numpy as np
from skimage.feature import graycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk

from ._validation import choices


GLCM_FEATURES = (
    "contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation",
    "entropy", "mean", "variance", "maximum_probability", "cluster_shade",
    "cluster_prominence",
)


@dataclass(frozen=True)
class TextureConfig:
    """Choose texture bands, spatial scales, directions, and quantization.

    Attributes:
        bands: Zero-based band indices; None uses the extraction's selected bands.
        features: GLCM feature names; an empty tuple disables GLCM calculations.
        distances: Positive integer pixel distances, reported separately.
        angles: Directions in radians, measured from increasing columns toward
            increasing rows. With angle_aggregation='none', output a0, a1, ...
            refer to positions in this tuple.
        levels: Number of uniformly spaced gray levels, between 2 and 256.
        value_range: Fixed (minimum, maximum), or None for each band's finite
            range over the entire included image. Never fitted per object.
        symmetric: Include both pair orders in each co-occurrence matrix.
        angle_aggregation: 'mean' averages available directions; 'none' retains
            each direction. A direction with no valid pixel pairs is NaN.
        entropy_radii: Positive integer radii for disk-shaped local entropy
            windows. Their mean and population standard deviation are reported.
    """

    bands: tuple[int, ...] | None = None
    features: tuple[str, ...] = GLCM_FEATURES[:7]
    distances: tuple[int, ...] = (1,)
    angles: tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    levels: int = 32
    value_range: tuple[float, float] | None = None
    symmetric: bool = True
    angle_aggregation: str = "mean"
    entropy_radii: tuple[int, ...] = ()

    def __post_init__(self):
        if self.bands is not None:
            object.__setattr__(self, "bands", tuple(self.bands))
        object.__setattr__(self, "features", choices(self.features, GLCM_FEATURES, "GLCM features"))
        for name in ("distances", "entropy_radii"):
            values = tuple(getattr(self, name))
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
                   for v in values):
                raise ValueError(f"{name} must contain positive integers in pixels.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates.")
            object.__setattr__(self, name, values)
        angles = tuple(float(a) for a in self.angles)
        if not angles or any(not np.isfinite(a) for a in angles) or len(set(angles)) != len(angles):
            raise ValueError("angles must contain distinct finite radians.")
        object.__setattr__(self, "angles", angles)
        if self.features and not self.distances:
            raise ValueError("GLCM features require at least one distance.")
        if isinstance(self.levels, (bool, np.bool_)) or not isinstance(self.levels, (int, np.integer)) or not 2 <= self.levels <= 256:
            raise ValueError("levels must be an integer between 2 and 256.")
        if self.angle_aggregation not in {"mean", "none"}:
            raise ValueError("angle_aggregation must be 'mean' or 'none'.")
        if not isinstance(self.symmetric, (bool, np.bool_)):
            raise TypeError("symmetric must be a boolean.")
        if self.value_range is not None:
            bounds = tuple(float(v) for v in self.value_range)
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError("value_range must contain finite minimum < maximum.")
            object.__setattr__(self, "value_range", bounds)


def texture_columns(config):
    columns = []
    for distance in config.distances:
        for feature in config.features:
            prefix = f"glcm_{feature}_d{distance}"
            if config.angle_aggregation == "mean":
                columns.append(prefix)
            else:
                columns.extend(f"{prefix}_a{i}" for i in range(len(config.angles)))
    for radius in config.entropy_radii:
        columns.extend([f"local_entropy_r{radius}_mean", f"local_entropy_r{radius}_std"])
    return columns


def quantize(values, valid, levels, bounds):
    result = np.zeros(values.shape, dtype=np.uint8)
    if valid.any() and bounds[1] > bounds[0]:
        scaled = (values[valid] - bounds[0]) / (bounds[1] - bounds[0])
        # Clipping before integer conversion avoids overflow for outliers.
        result[valid] = np.minimum(np.floor(np.clip(scaled, 0, 1) * levels), levels - 1).astype(np.uint8)
    return result


def _round_offset(value):
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _cooccurrence(image, valid, distance, angle, levels, symmetric):
    dy, dx = _round_offset(distance * np.sin(angle)), _round_offset(distance * np.cos(angle))
    height, width = image.shape
    matrix = np.zeros((levels, levels), dtype=float)
    if abs(dy) >= height or abs(dx) >= width:
        return matrix
    y0, y1 = max(0, -dy), min(height, height - dy)
    x0, x1 = max(0, -dx), min(width, width - dx)
    a = image[y0:y1, x0:x1]
    b = image[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    pairs = valid[y0:y1, x0:x1] & valid[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    encoded = a[pairs].astype(np.int64) * levels + b[pairs]
    matrix = np.bincount(encoded, minlength=levels * levels).reshape(levels, levels).astype(float)
    if symmetric:
        matrix += matrix.T.copy()
    total = matrix.sum()
    return matrix / total if total else matrix


def _properties(matrix, features):
    if not matrix.any():
        return {name: np.nan for name in features}
    result = {}
    i, j = np.indices(matrix.shape)
    mean_i, mean_j = (i * matrix).sum(), (j * matrix).sum()
    positive = matrix[matrix > 0]
    for name in features:
        if name in GLCM_FEATURES[:6]:
            result[name] = float(graycoprops(matrix[:, :, None, None], name)[0, 0])
        elif name == "entropy":
            result[name] = float(-(positive * np.log2(positive)).sum())
        elif name == "mean":
            result[name] = float(mean_i)
        elif name == "variance":
            result[name] = float(((i - mean_i) ** 2 * matrix).sum())
        elif name == "maximum_probability":
            result[name] = float(matrix.max())
        else:
            power = 3 if name == "cluster_shade" else 4
            result[name] = float(((i + j - mean_i - mean_j) ** power * matrix).sum())
    return result


def measure_texture(values, valid, config, bounds):
    result = dict.fromkeys(texture_columns(config), np.nan)
    valid = valid & np.isfinite(values)
    if not valid.any():
        return result
    image = quantize(values, valid, config.levels, bounds)
    for distance in config.distances:
        measurements = [_properties(_cooccurrence(image, valid, distance, angle, config.levels,
                                                  config.symmetric), config.features)
                        for angle in config.angles] if config.features else []
        for name in config.features:
            prefix = f"glcm_{name}_d{distance}"
            values_by_angle = np.array([m[name] for m in measurements])
            if config.angle_aggregation == "mean":
                finite = values_by_angle[np.isfinite(values_by_angle)]
                result[prefix] = float(finite.mean()) if finite.size else np.nan
            else:
                result.update({f"{prefix}_a{i}": value for i, value in enumerate(values_by_angle)})
    for radius in config.entropy_radii:
        local = entropy(image, disk(radius), mask=valid.astype(np.uint8))[valid]
        result[f"local_entropy_r{radius}_mean"] = float(local.mean())
        result[f"local_entropy_r{radius}_std"] = float(local.std())
    return result
