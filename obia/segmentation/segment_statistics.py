"""Compatibility entry points for raster feature extraction.

New workflows can use :func:`obia.features.extract_features` directly. The
existing functions keep their boolean flags and legacy output column names.
"""

from dataclasses import replace

import numpy as np

from obia.features import TextureConfig, extract_features
from obia.features._validation import band_indices
from obia.features.spectral import summarize, validate_statistics
from obia.features.texture import measure_texture


def _spectral_selection(calc_mean, calc_variance, calc_min, calc_max, calc_skewness, calc_kurtosis):
    return tuple(name for name, enabled in zip(
        ("mean", "variance", "min", "max", "skewness", "kurtosis"),
        (calc_mean, calc_variance, calc_min, calc_max, calc_skewness, calc_kurtosis)) if enabled)


def _texture_selection(calc_contrast, calc_dissimilarity, calc_homogeneity, calc_ASM, calc_energy, calc_correlation):
    return tuple(name for name, enabled in zip(
        ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"),
        (calc_contrast, calc_dissimilarity, calc_homogeneity, calc_ASM, calc_energy, calc_correlation)) if enabled)


def calculate_spectral_stats(
    image, statistics_bands, calc_mean=True, calc_variance=True, calc_min=True, calc_max=True,
    calc_skewness=True, calc_kurtosis=True, *, features=None, percentiles=(), mask=None,
):
    """Summarize a band-first (bands, height, width) array without modification.

    Legacy flags select the original statistics; ``features`` overrides them
    with names from ``SPECTRAL_FEATURES``. ``percentiles`` adds linear percentiles.
    Nonfinite and masked samples are excluded. A spatial ``mask`` defines the
    object footprint and denominator for ``valid_fraction``.
    """
    data = np.ma.asarray(image, dtype=float).filled(np.nan)
    if data.ndim != 3:
        raise ValueError("image must have shape (bands, height, width).")
    bands = band_indices(statistics_bands, data.shape[0])
    if features is None:
        features = _spectral_selection(calc_mean, calc_variance, calc_min, calc_max, calc_skewness, calc_kurtosis)
    features, percentiles = validate_statistics(features, percentiles)
    inside = np.ones(data.shape[1:], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if inside.shape != data.shape[1:]:
        raise ValueError("mask must have shape (height, width).")
    result = {}
    for band in bands:
        stats = summarize(data[band][inside], int(inside.sum()), features, percentiles)
        result.update({f"b{band}_{name}": value for name, value in stats.items()})
    return result


def calculate_textural_stats(
    image, textural_bands, calc_contrast=True, calc_dissimilarity=True,
    calc_homogeneity=True, calc_ASM=True, calc_energy=True, calc_correlation=True,
    *, texture_config=None, mask=None, channel_axis=0,
):
    """Calculate texture from a band-first array (or explicit channel_axis=-1).

    Only pixel pairs with both endpoints inside ``mask`` and finite are counted.
    Without ``texture_config``, legacy bN_contrast etc. columns use distance 2,
    four directions, 256 gray levels, and the valid band range of this array.
    An explicit ``TextureConfig`` enables additional features and scale columns.
    Use ``extract_features`` to fit quantization ranges once across all objects.
    """
    data = np.ma.asarray(image, dtype=float).filled(np.nan)
    if data.ndim != 3 or channel_axis not in (0, -1, 2):
        raise ValueError("image must be 3D; channel_axis must be 0 or -1 (2).")
    data = np.moveaxis(data, channel_axis, 0)
    bands = band_indices(textural_bands, data.shape[0])
    inside = np.ones(data.shape[1:], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if inside.shape != data.shape[1:]:
        raise ValueError("mask must have shape (height, width).")
    legacy = texture_config is None
    if legacy:
        features = _texture_selection(calc_contrast, calc_dissimilarity, calc_homogeneity,
                                      calc_ASM, calc_energy, calc_correlation)
        texture_config = TextureConfig(features=features, distances=(2,), levels=256)
    elif not isinstance(texture_config, TextureConfig):
        raise TypeError("texture_config must be a TextureConfig.")
    result = {}
    for band in bands:
        valid = inside & np.isfinite(data[band])
        values = data[band][valid]
        bounds = texture_config.value_range or ((float(values.min()), float(values.max()))
                                               if values.size else (0.0, 0.0))
        stats = measure_texture(data[band], valid, texture_config, bounds)
        for name, value in stats.items():
            if legacy:
                name = name.removeprefix("glcm_").removesuffix("_d2")
            result[f"b{band}_{name}"] = value
    return result


def create_objects(
    segments, image, spectral_bands=None, textural_bands=None,
    calculate_spectral=True, calculate_textural=True,
    calc_mean=True, calc_variance=True, calc_min=True, calc_max=True, calc_skewness=True, calc_kurtosis=True,
    calc_contrast=True, calc_dissimilarity=True, calc_homogeneity=True, calc_ASM=True, calc_energy=True,
    calc_correlation=True, *, spectral_features=None, percentiles=(), indices=None,
    shape_features=(), shape_units="map", texture_config=None, mask=None,
):
    """Enrich segment polygons with selected raster features.

    Existing flags and bN_mean/bN_contrast-style columns are retained. New
    ``spectral_features``, ``percentiles``, ``indices``, ``shape_features``,
    ``shape_units``, ``texture_config`` and ``mask`` options correspond to
    ``extract_features``. All bands are numeric zero-based indices into the
    loaded image. ``calculate_spectral=False`` disables band/index distributions;
    ``calculate_textural=False`` disables texture. Shape can be requested alone.
    """
    features = spectral_features
    if features is None:
        features = _spectral_selection(calc_mean, calc_variance, calc_min, calc_max, calc_skewness, calc_kurtosis)
    if not calculate_spectral:
        features, percentiles = (), ()
    legacy_texture = texture_config is None
    if calculate_textural:
        if legacy_texture:
            requested = _texture_selection(calc_contrast, calc_dissimilarity, calc_homogeneity,
                                           calc_ASM, calc_energy, calc_correlation)
            texture_config = TextureConfig(features=requested, distances=(2,), levels=256)
        elif not isinstance(texture_config, TextureConfig):
            raise TypeError("texture_config must be a TextureConfig.")
        if texture_config.bands is None:
            texture_config = replace(texture_config, bands=band_indices(textural_bands, image.img_data.shape[2]))
    else:
        texture_config = None
    result = extract_features(
        segments, image, bands=spectral_bands, spectral=features, percentiles=percentiles,
        indices=indices, shape=shape_features, shape_units=shape_units, texture=texture_config, mask=mask,
    )
    if legacy_texture and texture_config is not None:
        rename = {f"b{b}_glcm_{name}_d2": f"b{b}_{name}"
                  for b in texture_config.bands for name in texture_config.features}
        # Re-extraction updates legacy columns, preserving unrelated attributes.
        result = result.drop(columns=[name for name in rename.values() if name in result], errors="ignore")
        result = result.rename(columns=rename)
    return result
