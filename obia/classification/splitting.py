"""Inspectable object holdouts and cross-validation folds.

Choose groups and spatial scales to match the intended prediction task; spatial
cross-validation is not a substitute for probability sampling of map accuracy.
See Roberts et al. (2017), doi:10.1111/ecog.02881, and Wadoux et al. (2021),
doi:10.1016/j.ecolmodel.2021.109692.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Hashable, Literal, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from shapely import STRtree, distance
from shapely.geometry import box
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold, StratifiedKFold,
    train_test_split,
)
from sklearn.utils.multiclass import check_classification_targets


def _labels(data):
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        raise ValueError("Labelled objects must be a DataFrame with unique column names.")
    if "feature_class" not in data or not len(data):
        raise ValueError("training_classes must contain labelled rows and a feature_class column.")
    labels = data["feature_class"]
    if labels.isna().any():
        raise ValueError("Training labels cannot be missing; remove unlabelled rows before splitting.")
    strings = [isinstance(value, str) for value in labels]
    if any(strings) and not all(strings):
        raise ValueError("Training labels must be all strings or all numeric class codes.")
    y = np.asarray(labels.tolist())
    check_classification_targets(y)
    return y


def _identity(data, group_columns):
    """Track row identity without binding a split to a particular feature set."""
    result = pd.DataFrame(index=data.index.copy())
    for column in dict.fromkeys(("feature_class", "segment_id", *group_columns)):
        if column in data:
            result[("column", column)] = data[column].to_numpy(copy=True)
    if isinstance(data, gpd.GeoDataFrame):
        result[("geometry", "wkb")] = data.geometry.to_wkb().to_numpy()
        result.attrs["crs"] = data.crs
    return result


@dataclass(frozen=True)
class ObjectSplit:
    """A reusable partition of a labelled object table, created by split helpers.

    Attributes:
        train_indices: Zero-based row positions for fitting, suitable for ``iloc``.
        test_indices: Zero-based row positions for evaluation.
        excluded_indices: Positions omitted from both subsets, including buffers.
        assignments: Copy of an inspectable table with ``row_position``, ``subset``
            (train/test/excluded), labels, IDs when present, and group/block columns.
            Geometry and CRS are retained when the source is a GeoDataFrame.
        class_counts: Counts of each class in train, test, and excluded subsets.
        metadata: Method, settings, and achieved sample sizes. Distances and grid
            sizes use projected CRS units, not necessarily metres.
        feature_exclusions: Grouping column names, excluded from model predictors.

    Indices are positional even when the source index contains duplicates. A
    split can be reused after changing feature values/columns, but its index,
    labels, IDs, geometry, CRS, and named group columns must remain unchanged.
    """

    _train: tuple[int, ...] = field(repr=False)
    _test: tuple[int, ...] = field(repr=False)
    _excluded: tuple[int, ...] = field(repr=False)
    _assignments: pd.DataFrame = field(repr=False, compare=False)
    _source: pd.DataFrame = field(repr=False, compare=False)
    _metadata: dict = field(repr=False, compare=False)
    feature_exclusions: tuple = ()

    @property
    def train_indices(self) -> np.ndarray:
        return np.asarray(self._train, dtype=np.intp)

    @property
    def test_indices(self) -> np.ndarray:
        return np.asarray(self._test, dtype=np.intp)

    @property
    def excluded_indices(self) -> np.ndarray:
        return np.asarray(self._excluded, dtype=np.intp)

    @property
    def assignments(self) -> pd.DataFrame:
        return self._assignments.copy(deep=True)

    @property
    def class_counts(self) -> pd.DataFrame:
        counts = pd.crosstab(self._assignments.feature_class, self._assignments.subset)
        return counts.reindex(columns=["train", "test", "excluded"], fill_value=0)

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()

    def as_sklearn(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(train_indices, test_indices)`` for scikit-learn's ``cv`` iterable."""
        return self.train_indices, self.test_indices

    def validate_for(self, data: pd.DataFrame) -> None:
        """Reject reuse on reordered or different objects; feature changes are allowed."""
        current = _identity(data, self.feature_exclusions)
        if not current.equals(self._source) or current.attrs != self._source.attrs:
            raise ValueError(
                "The split does not match these labelled objects or their row order. "
                "Recreate the split after changing the index, labels, IDs, groups, geometry, or CRS."
            )


def _projected(data):
    if not isinstance(data, gpd.GeoDataFrame) or data.crs is None or not data.crs.is_projected:
        raise ValueError("Spatial blocks and buffers require a GeoDataFrame with a projected CRS.")
    geometry = data.geometry
    if (geometry.isna().any() or geometry.is_empty.any() or not geometry.is_valid.all()
            or not np.isfinite(geometry.bounds.to_numpy()).all()):
        raise ValueError("Spatial splitting requires finite, nonempty, valid geometries.")


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number in CRS units.")
    return float(value)


def _groups(data, method, groups, block_size, block_origin):
    extras, exclusions = {}, ()
    if method == "spatial":
        if groups is not None:
            raise ValueError("Use either spatial blocks or supplied groups, not both.")
        _projected(data)
        size = _nonnegative(block_size, "block_size")
        if size == 0:
            raise ValueError("block_size must be greater than zero.")
        try:
            origin = np.asarray(block_origin, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("block_origin must contain two finite coordinates.") from exc
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("block_origin must contain two finite coordinates.")
        points = data.geometry.representative_point()
        cells = np.floor((np.column_stack((points.x, points.y)) - origin) / size)
        if not np.isfinite(cells).all() or (np.abs(cells) >= np.iinfo(np.int64).max).any():
            raise ValueError("block_size is too small for these coordinates.")
        cells = cells.astype(np.int64)
        extras = {"block_x": cells[:, 0], "block_y": cells[:, 1]}
        values = pd.Series([f"{x}:{y}" for x, y in cells], index=data.index)
    else:
        if block_size is not None:
            raise ValueError("block_size is only used with method='spatial'.")
        if groups is None:
            if method == "grouped":
                raise ValueError("method='grouped' requires groups: a column name or one group per row.")
            return None, None, extras, exclusions
        if method == "random":
            raise ValueError("Use method='grouped' to keep supplied groups together.")
        if isinstance(groups, str):
            if groups not in data or groups == "feature_class" or (
                isinstance(data, gpd.GeoDataFrame) and groups == data.geometry.name
            ):
                raise ValueError("groups must name an existing non-label, non-geometry column.")
            exclusions = (groups,)
            values = data[groups]
        else:
            if isinstance(groups, pd.Series) and not groups.index.equals(data.index):
                raise ValueError("A groups Series must have the same index and row order as the objects.")
            try:
                values = pd.Series(list(groups), index=data.index, dtype=object)
            except (TypeError, ValueError) as exc:
                raise ValueError("groups must supply exactly one hashable group per row.") from exc
    if values.isna().any():
        raise ValueError("Group labels cannot be missing.")
    try:
        codes, _ = pd.factorize(values, sort=False)
    except TypeError as exc:
        raise ValueError("Group labels must be hashable scalar values.") from exc
    return codes, values.to_numpy(copy=True), extras, exclusions


def _positions(values, n, name):
    if values is None:
        raise ValueError(f"method='explicit' requires {name}.")
    positions = np.asarray(values)
    if positions.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of row positions or a boolean mask.")
    if positions.dtype.kind == "b":
        if len(positions) != n:
            raise ValueError(f"The {name} boolean mask must have one entry per row.")
        positions = np.flatnonzero(positions)
    if not len(positions) or positions.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain nonempty integer row positions.")
    if (positions < 0).any() or (positions >= n).any() or len(np.unique(positions)) != len(positions):
        raise ValueError(f"{name} must contain distinct row positions within the object table.")
    return positions.astype(np.intp)


def _buffer_train(data, train, test, buffer_distance, codes):
    if buffer_distance == 0 or not len(test):
        return train
    # Query expanded bounding boxes, then check exact geometry distances. This
    # avoids centroid leakage and buffer-polygon approximation at curved edges.
    geometry = data.geometry.iloc[train].to_numpy()
    tree = STRtree(geometry)
    removed = np.zeros(len(train), dtype=bool)
    for test_geometry in data.geometry.iloc[test]:
        xmin, ymin, xmax, ymax = test_geometry.bounds
        candidates = tree.query(box(xmin - buffer_distance, ymin - buffer_distance,
                                    xmax + buffer_distance, ymax + buffer_distance))
        removed[candidates[distance(geometry[candidates], test_geometry) <= buffer_distance]] = True
    if codes is not None:
        # A group remains a single unit even when only one member hits the buffer.
        removed = np.isin(codes[train], codes[train[removed]])
    return train[~removed]


def _coverage(y, train, test, require_test_classes):
    if not len(train):
        raise ValueError("The split leaves no training objects; reduce the buffer or change the partition.")
    missing_train = set(y) - set(y[train])
    if missing_train:
        raise ValueError(f"The training split is missing classes: {sorted(missing_train)}. "
                         "Change the partition, reduce the buffer, or collect more labelled groups.")
    if require_test_classes and len(test) and set(y[test]) != set(y):
        raise ValueError("The test split is missing classes; change the partition or use stratify=False.")


def _make_split(data, y, train, test, codes, group_values, extras, exclusions, metadata, *, require_test_classes):
    if np.intersect1d(train, test).size:
        raise ValueError("Training and test row positions must not overlap.")
    if codes is not None and np.intersect1d(codes[train], codes[test]).size:
        raise ValueError("A group cannot occur in both training and test subsets.")
    train = _buffer_train(data, train, test, metadata["buffer_distance"], codes)
    _coverage(y, train, test, require_test_classes)
    subsets = np.full(len(data), "excluded", dtype=object)
    subsets[train], subsets[test] = "train", "test"
    excluded = np.flatnonzero(subsets == "excluded")
    assignments = pd.DataFrame({"row_position": np.arange(len(data)), "subset": subsets,
                                "feature_class": y, **extras}, index=data.index.copy())
    if "segment_id" in data:
        assignments["segment_id"] = data.segment_id.to_numpy(copy=True)
    if group_values is not None:
        assignments["group"] = group_values
    if isinstance(data, gpd.GeoDataFrame):
        assignments = gpd.GeoDataFrame(assignments, geometry=data.geometry.to_numpy(), crs=data.crs)
    metadata = {**metadata, "n_train": len(train), "n_test": len(test), "n_excluded": len(excluded)}
    return ObjectSplit(tuple(train), tuple(test), tuple(excluded), assignments,
                       _identity(data, exclusions), metadata, exclusions)


def split_objects(
    data: pd.DataFrame, *, method: Literal["random", "grouped", "spatial", "explicit"] = "random",
    test_size: float | int | None = 0.2, random_state: int | None = 42, stratify: bool = True,
    groups: str | Sequence[Hashable] | pd.Series | np.ndarray | None = None,
    block_size: float | None = None, block_origin: tuple[float, float] = (0.0, 0.0),
    buffer_distance: float = 0.0, train_indices: ArrayLike | None = None,
    test_indices: ArrayLike | None = None, n_candidates: int = 128,
) -> ObjectSplit:
    """Create a random, grouped, spatial, or explicitly assigned object holdout.

    Args:
        data: Labelled DataFrame/GeoDataFrame with ``feature_class``. Spatial
            methods/buffers require valid geometry and a projected CRS.
        method: ``random``, ``grouped``, ``spatial``, or ``explicit``.
        test_size: Fraction or integer count of rows for random splits, or of
            groups/occupied blocks for grouped/spatial splits. None fits all rows.
            Ignored for explicit partitions, whose indices determine their size.
        random_state: Seed for reproducible holdout selection.
        stratify: Random splits use class stratification. Grouped/spatial splits
            search candidate group partitions for approximate class balance with
            all classes in both subsets. Explicit splits retain supplied subsets
            and check test-class coverage when True. Training always needs all classes.
        groups: Group column name or row-ordered labels; required for grouped
            splits and optional for enforcing group separation in explicit splits.
            A named group column is excluded from classifier features.
        block_size: Positive square-grid width in projected CRS units, required
            for spatial splits. Each object's representative point selects its
            block, and the entire object stays in that block.
        block_origin: Fixed grid origin (x, y), default (0, 0).
        buffer_distance: Exclude training geometries at distance <= this value
            from any test geometry. Zero disables buffering. If groups/blocks
            are used, exclude the entire affected training group/block.
        train_indices: Explicit zero-based row positions or a boolean mask.
        test_indices: Explicit zero-based row positions or a boolean mask.
            Rows in neither subset are marked excluded.
        n_candidates: Number of random group partitions considered for stratified
            grouped/spatial holdouts. The search is approximate and can fail even
            if a feasible partition exists; increase this or supply explicit indices.

    Returns:
        ObjectSplit: Reusable indices, map-ready assignments, and class counts.
            Positive buffers can reduce training size and alter class balance.
    """
    if method not in {"random", "grouped", "spatial", "explicit"}:
        raise ValueError("method must be 'random', 'grouped', 'spatial', or 'explicit'.")
    if not isinstance(stratify, (bool, np.bool_)):
        raise ValueError("stratify must be True or False.")
    y = _labels(data)
    buffer_distance = _nonnegative(buffer_distance, "buffer_distance")
    if buffer_distance:
        _projected(data)
    codes, values, extras, exclusions = _groups(data, method, groups, block_size, block_origin)
    metadata = dict(method=method, test_size=test_size, random_state=random_state,
                    stratify=bool(stratify), buffer_distance=buffer_distance)
    if method == "spatial":
        metadata.update(block_size=float(block_size), block_origin=tuple(map(float, block_origin)))
    positions = np.arange(len(data))
    if method == "explicit":
        train = _positions(train_indices, len(data), "train_indices")
        test = _positions(test_indices, len(data), "test_indices")
        metadata["test_size"] = None
    else:
        if train_indices is not None or test_indices is not None:
            raise ValueError("train_indices and test_indices require method='explicit'.")
        if test_size is None:
            train, test = positions, np.array([], dtype=np.intp)
        elif isinstance(test_size, (bool, np.bool_)):
            raise ValueError("test_size must be a fraction, an integer count, or None.")
        elif codes is None:
            try:
                train, test = train_test_split(positions, test_size=test_size, random_state=random_state,
                                               stratify=y if stratify else None)
            except ValueError as exc:
                raise ValueError(f"Cannot create the requested holdout split: {exc} "
                                 "Add labelled examples, adjust test_size, or use test_size=None.") from exc
        else:
            if isinstance(n_candidates, (bool, np.bool_)) or not isinstance(n_candidates, Integral) or n_candidates < 1:
                raise ValueError("n_candidates must be a positive integer.")
            splitter = GroupShuffleSplit(n_splits=n_candidates if stratify else 1,
                                        test_size=test_size, random_state=random_state)
            metadata["n_candidates"] = n_candidates if stratify else 1
            if stratify:
                _, label_codes = np.unique(y, return_inverse=True)
                proportions = np.bincount(label_codes) / len(y)

                def imbalance(pair):
                    return sum(np.abs(np.bincount(label_codes[part], minlength=len(proportions))
                                      / len(part) - proportions).sum() for part in pair)

            # Keep only the best viable partition, rather than materializing
            # n_candidates copies of the full object's row indices in memory.
            best, best_score = None, np.inf
            for train, test in splitter.split(positions, y, codes):
                score = imbalance((train, test)) if stratify else 0.0
                if score >= best_score:
                    continue
                try:
                    candidate = _make_split(data, y, train, test, codes, values, extras, exclusions,
                                            metadata, require_test_classes=stratify)
                except ValueError:
                    if not stratify:
                        raise
                else:
                    best, best_score = candidate, score
                    if score == 0:
                        break
            if best is not None:
                return best
            raise ValueError(
                "No class-complete grouped split was found. Classes need multiple independent groups. "
                "Adjust test_size, block_size, buffer_distance or n_candidates; collect more groups, "
                "or supply an explicit partition. stratify=False permits absent test classes."
            )
    return _make_split(data, y, train, test, codes, values, extras, exclusions,
                       metadata, require_test_classes=stratify)


def cross_validation_splits(
    data: pd.DataFrame, *, method: Literal["random", "grouped", "spatial"] = "random",
    n_splits: int = 5, random_state: int | None = 42, stratify: bool = True,
    groups: str | Sequence[Hashable] | pd.Series | np.ndarray | None = None,
    block_size: float | None = None, block_origin: tuple[float, float] = (0.0, 0.0),
    buffer_distance: float = 0.0,
) -> list[ObjectSplit]:
    """Create reusable random, grouped, or spatial cross-validation folds.

    Each object is tested once. Buffers remove training objects separately for
    each fold; test objects are never removed. Arguments have the same meaning
    as in ``split_objects``. Grouped/spatial stratification uses scikit-learn's
    StratifiedGroupKFold and is approximate. Non-stratified group folds use
    GroupKFold's deterministic allocation (random_state has no effect there).

    All folds require every class in training; stratify=True also requires every
    class in each test fold. Insufficient class/group coverage raises an error,
    rather than silently moving objects between groups. Reduce n_splits or change
    the partition when necessary. No models are fitted by this function.

    Returns:
        list[ObjectSplit]: Pass each split to ``classify(..., split=fold)`` or use
            ``[fold.as_sklearn() for fold in folds]`` as scikit-learn's cv argument.
            Generate folds only from development data after reserving final tests.
    """
    if method not in {"random", "grouped", "spatial"}:
        raise ValueError("Cross-validation method must be 'random', 'grouped', or 'spatial'.")
    if isinstance(n_splits, (bool, np.bool_)) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")
    if not isinstance(stratify, (bool, np.bool_)):
        raise ValueError("stratify must be True or False.")
    y = _labels(data)
    buffer_distance = _nonnegative(buffer_distance, "buffer_distance")
    if buffer_distance:
        _projected(data)
    codes, values, extras, exclusions = _groups(data, method, groups, block_size, block_origin)
    if codes is None:
        cls = StratifiedKFold if stratify else KFold
        splitter = cls(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif stratify:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        splitter = GroupKFold(n_splits=n_splits)
    metadata = dict(method=method, n_splits=n_splits, random_state=random_state,
                    stratify=bool(stratify), buffer_distance=buffer_distance)
    if method == "spatial":
        metadata.update(block_size=float(block_size), block_origin=tuple(map(float, block_origin)))
    folds = []
    for fold, (train, test) in enumerate(splitter.split(np.arange(len(data)), y, codes)):
        try:
            folds.append(_make_split(data, y, train, test, codes, values, extras, exclusions,
                                     {**metadata, "fold": fold}, require_test_classes=stratify))
        except ValueError as exc:
            raise ValueError(f"Invalid cross-validation fold {fold}: {exc}") from exc
    return folds
