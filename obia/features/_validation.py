"""Shared validation for raster feature requests."""

import numpy as np


def band_indices(bands, count):
    values = tuple(range(count)) if bands is None else tuple(bands)
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("Bands must be zero-based integer indices into image.img_data.")
        if not 0 <= value < count:
            raise IndexError(f"Band index {value} out of range for {count} loaded bands.")
    if len(set(values)) != len(values):
        raise ValueError("Band indices must not be repeated.")
    return tuple(int(value) for value in values)


def choices(values, available, name):
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of names, not a string.")
    values = tuple(values)
    unknown = set(values) - set(available)
    if unknown:
        raise ValueError(f"Unknown {name}: {sorted(unknown)}. Choose from {available}.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")
    return values


def number_suffix(value):
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")
