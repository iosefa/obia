"""Spectral distributions and indices with explicit numeric band selection."""

import numpy as np
from scipy.stats import kurtosis, skew

from ._validation import band_indices, choices, number_suffix


SPECTRAL_FEATURES = (
    "mean", "std", "variance", "min", "max", "median", "iqr", "mad",
    "skewness", "kurtosis", "valid_count", "valid_fraction",
)
_NORMALIZED_DIFFERENCE = (
    "normalized_difference", "ndvi", "gndvi", "ndre", "ndwi", "mndwi",
    "ndmi", "nbr", "ndbi",
)
SPECTRAL_INDICES = _NORMALIZED_DIFFERENCE + ("ratio", "difference", "exg", "vari", "brightness")


def validate_statistics(features, percentiles):
    features = choices(features, SPECTRAL_FEATURES, "spectral features")
    percentiles = tuple(float(p) for p in percentiles)
    if any(not np.isfinite(p) or not 0 <= p <= 100 for p in percentiles):
        raise ValueError("percentiles must be finite numbers between 0 and 100.")
    if len({number_suffix(p) for p in percentiles}) != len(percentiles):
        raise ValueError("percentiles must have distinct output names.")
    return features, percentiles


def summarize(values, total, features, percentiles):
    """Summarize finite samples; total includes invalid or masked object pixels."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    result = {name: np.nan for name in features}
    result.update({f"p{number_suffix(p)}": np.nan for p in percentiles})
    count = values.size
    if "valid_count" in features:
        result["valid_count"] = int(count)
    if "valid_fraction" in features:
        result["valid_fraction"] = count / total if total else np.nan
    if not count:
        return result
    functions = {
        "mean": np.mean, "std": np.std, "variance": np.var,
        "min": np.min, "max": np.max, "median": np.median,
        "iqr": lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        "mad": lambda x: np.median(np.abs(x - np.median(x))),
    }
    for name in features:
        if name in functions:
            result[name] = float(functions[name](values))
    # Standardized higher moments are undefined for constant samples.
    if np.ptp(values) > 0:
        if "skewness" in features:
            result["skewness"] = float(skew(values, bias=True))
        if "kurtosis" in features:
            result["kurtosis"] = float(kurtosis(values, fisher=True, bias=True))
    if percentiles:
        for p, value in zip(percentiles, np.percentile(values, percentiles, method="linear")):
            result[f"p{number_suffix(p)}"] = float(value)
    return result


def validate_index(method, bands, count):
    if method not in SPECTRAL_INDICES:
        raise ValueError(f"Unknown spectral index {method!r}. Choose from {SPECTRAL_INDICES}.")
    if bands is None:
        raise TypeError("Spectral indices require explicit numeric band indices in formula order.")
    bands = band_indices(bands, count)
    required = 3 if method in {"exg", "vari"} else 2
    if method == "brightness":
        if not bands:
            raise ValueError("brightness requires at least one band index.")
    elif len(bands) != required:
        raise ValueError(f"{method} requires exactly {required} band indices in formula order.")
    return bands


def compute_index(data: np.ndarray, method: str, bands: tuple[int, ...]) -> np.ndarray:
    """Calculate a spectral index from explicit zero-based band indices.

    Args:
        data: Raster array shaped (height, width, bands), optionally masked.
        method: Name in ``SPECTRAL_INDICES``. Normalized-difference methods use
            (a-b)/(a+b), ratio uses a/b, and difference uses a-b. ``exg`` uses
            (2*b-a-c)/(a+b+c), ``vari`` uses (b-a)/(b+a-c), and brightness is
            the arithmetic mean. Here a, b, c are the bands in supplied order.
        bands: Ordered numeric band indices. No wavelengths are inferred.
            For example, ndvi with (3, 0) calculates (b3-b0)/(b3+b0).

    Returns:
        np.ndarray: A float64 index image. Masked/nonfinite operands, zero
            denominators, and nonfinite results become NaN. Input is unchanged.
    """
    data = np.ma.asarray(data, dtype=float).filled(np.nan)
    if data.ndim != 3:
        raise ValueError("data must have shape (height, width, bands).")
    bands = validate_index(method, bands, data.shape[2])
    selected = data[:, :, bands]
    valid = np.isfinite(selected).all(axis=2)
    a = selected[:, :, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if method == "brightness":
            result = selected.mean(axis=2)
        else:
            b = selected[:, :, 1]
            if method in _NORMALIZED_DIFFERENCE:
                result = (a - b) / (a + b)
            elif method == "ratio":
                result = a / b
            elif method == "difference":
                result = a - b
            elif method == "exg":
                c = selected[:, :, 2]
                result = (2 * b - a - c) / (a + b + c)
            else:  # VARI
                c = selected[:, :, 2]
                result = (b - a) / (b + a - c)
    return np.where(valid & np.isfinite(result), result, np.nan)
