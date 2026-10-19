# Training and Validation

Choose a split that represents where the model will be used. For predictions in
new fields, sites, or regions, hold out those units together. Nearby objects can
share environmental conditions, acquisition effects, and labels, so a random
object split can give an optimistic estimate of performance in new areas.
Validation should reflect the prediction task and dependence in the data
([Roberts et al., 2017](https://doi.org/10.1111/ecog.02881)).

Use `split_objects()` to assign labelled objects to training, test, and excluded
subsets before fitting. It returns an `ObjectSplit` you can inspect and reuse
across classifiers. Without a supplied split, `classify()` uses a stratified
random 80/20 split.

## Choose what to hold out

| Intended use | Starting point |
| --- | --- |
| New fields, forest stands, survey sites, or scenes | `method="grouped"`, using the real independent sampling unit. |
| Unsampled areas within a region, without natural groups | `method="spatial"`, with a block size appropriate to the deployment distance and spatial dependence. |
| A predefined region, separate survey, or future acquisition | `method="explicit"`, with the train and test rows assigned by the study design. |
| Interpolation among sampled objects, or a quick development check | `method="random"`; nearby training and test objects may be strongly related. |

Keep every segment belonging to the same underlying field or survey unit in one
group. Repeated observations of a site should remain together when evaluating
transfer to new sites. For future-date prediction, assign earlier observations
to training and later ones to testing explicitly; random date groups do not
enforce time order.

The square-grid method holds out occupied cells distributed across the scene.
It does not automatically reserve one contiguous region. Use explicit partitions
or region IDs when the target is a separate region. Block size and buffer distance
are study choices, not guarantees of independence. Consider the scale of spatial
dependence and distances to training data expected at deployment; there is no
universal distance in metres. Geographic separation alone also cannot establish
whether new conditions resemble the training data
([Meyer & Pebesma, 2021](https://doi.org/10.1111/2041-210X.13650)).

## Hold out groups

The examples use `labelled`, a DataFrame or GeoDataFrame with a `feature_class`
column, and `objects`, the polygons to classify. Both need the same feature
columns. For this grouped example, `labelled` also needs a `site_id` column
identifying the site or field each object belongs to.

```python
from obia.classification import classify, split_objects

partition = split_objects(
    labelled,
    method="grouped",
    groups="site_id",
    test_size=0.2,
    random_state=42,
)
print(partition.class_counts)
print(partition.metadata)

result = classify(
    objects, labelled,
    method="rf",
    split=partition,
    feature_columns=["b0_mean", "b1_mean", "b2_mean"],
    n_estimators=300,
    random_state=42,
    compute_reports=True,
)
print(result.report)
```

For grouped and spatial holdouts, `test_size` is a fraction or integer count of
**groups or occupied blocks**, following
[GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
It is not a requested fraction of objects or land area. Unequal group sizes can
produce a very different object fraction; inspect the achieved counts.

`stratify=True` searches up to `n_candidates=128` seeded group partitions, favoring
class proportions close to the whole table before buffering. It requires every
class in both train and test and keeps groups intact. This is an approximate
search: it can fail even when a feasible partition exists. Increase
`n_candidates`, adjust the design, or supply explicit assignments. Do not break
groups simply to obtain balanced classes. `stratify=False` permits absent test
classes; the training subset must still contain every class in the input table.

A named grouping column is automatically excluded from model predictors, even
if it is numeric. `groups` also accepts one group label per row; a pandas Series
must have the same index and row order as `labelled`. When supplying an array,
select `feature_columns` explicitly to exclude any numeric group metadata that
also remains in the table.

## Hold out spatial blocks

```python
# Use a projected CRS appropriate to the study area before creating the split.
partition = split_objects(
    labelled,
    method="spatial",
    block_size=1000,
    block_origin=(0, 0),
    buffer_distance=100,
    test_size=0.2,
    random_state=42,
)
partition.assignments.plot(column="subset", categorical=True, legend=True)
print(partition.class_counts)
```

The example distances are illustrative. Both values use the projected CRS's
units, which are not necessarily metres. Geographic coordinates, missing CRS,
and invalid, missing, or empty geometries are rejected for spatial splitting or
positive buffers. Choose a suitable projected CRS and repair geometry first;
OBIA does not choose a projection automatically.

Blocks are squares on a fixed grid. An interior representative point assigns
each object to one block, and the entire object stays together even if it crosses
a block boundary. `block_origin` fixes the grid alignment; it defaults to `(0, 0)`.

A positive `buffer_distance` removes training geometries whose minimum distance
to any test geometry is **less than or equal to** that distance. Distances are
measured between the actual geometries, not their centroids. If one member of a
training group or block meets the buffer, the whole training group or block is
excluded. Test objects are retained. This can remove substantial training data,
especially among contiguous segments, and can change class balance. Zero disables
buffering, including for touching objects. Buffers also work with grouped,
random, and explicit splits when projected geometry is available.

Review the map and class counts before fitting. OBIA raises an error if buffering
leaves no training objects or removes a training class. With `stratify=True`,
grouped holdouts try other candidate partitions; cross-validation reports the
invalid fold instead of silently skipping it.

## Inspect and reuse assignments

Every partition exposes:

- `train_indices`, `test_indices`, and `excluded_indices`: zero-based row
  positions for `labelled.iloc[...]`, independent of the DataFrame's index labels.
- `assignments`: a copy containing `row_position`, `subset`, `feature_class`,
  optional `segment_id`, group/block IDs, and geometry and CRS when available.
- `class_counts`: object counts per class in each subset, including exclusions.
- `metadata`: split settings and achieved subset sizes.

```python
training_objects = labelled.iloc[partition.train_indices]
test_objects = labelled.iloc[partition.test_indices]
partition.assignments.to_file("validation_split.gpkg", driver="GPKG")

# The same held-out objects are used for another model.
other = classify(objects, labelled, method="extra_trees", split=partition,
                 feature_columns=["b0_mean", "b1_mean", "b2_mean"],
                 random_state=42, compute_reports=True)
```

`result.split` retains the partition used, including for an automatically created
random split. A supplied `split` overrides `classify()`'s `test_size`,
`split_random_state`, and `stratify` settings. Only training rows fit the model
and its preprocessing; test rows supply reports, and excluded rows supply neither.
The prediction table can still contain all objects.

Reuse requires the same source row order, index, labels, IDs, named groups, and
geometry/CRS. Changing feature columns or values is allowed. Create a new split
after filtering or reordering the labelled table; duplicate index labels do not
change the positional meaning of the indices. Retain stable object IDs when
working with tables that lack geometry or unique index labels.

## Use predefined training and test sets

Explicit assignments accept integer row positions or boolean masks. Rows assigned
to neither subset are excluded. Train and test cannot overlap. Supplying `groups`
also enforces group separation.

```python
partition = split_objects(
    labelled,
    method="explicit",
    train_indices=labelled["survey_role"].eq("development").to_numpy(),
    test_indices=labelled["survey_role"].eq("test").to_numpy(),
    groups="site_id",
)
```

For independently collected training and test tables, combine them in a known
order and assign their positions explicitly. Use the same feature definitions,
label codes, and CRS for both tables:

```python
import numpy as np
import pandas as pd

labelled = pd.concat([training_table, test_table], ignore_index=True)
partition = split_objects(
    labelled,
    method="explicit",
    train_indices=np.arange(len(training_table)),
    test_indices=np.arange(len(training_table), len(labelled)),
)
result = classify(objects, labelled, method="rf", split=partition,
                  feature_columns=["b0_mean", "b1_mean", "b2_mean"],
                  compute_reports=True, random_state=42)
```

Explicit partitions are never reshuffled for balance. All input classes must be
represented in training. Set `stratify=False` if a predefined test set lacks some
classes; their test performance cannot then be estimated from that set. The
caller remains responsible for ensuring that separate tables do not contain the
same underlying objects or sampling units.

## Compare models with cross-validation

`cross_validation_splits()` returns inspectable `ObjectSplit` folds for random,
grouped, or spatial validation. Each object appears in a test subset exactly once;
buffer exclusions apply separately to each fold's training subset. Stratified
group folds use
[StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
which approximates class balance while preserving groups. With `stratify=False`,
grouped/spatial folds use deterministic `GroupKFold`; `random_state` has no effect
in that case. Every fold must retain all training classes, and stratified folds
also require all test classes.

Reserve a final test set before choosing model settings. Use cross-validation on
the remaining development data to compare those settings, then evaluate the
selected model on the untouched test set. Build inner folds using only development
rows, with the same independence assumptions as the outer split:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from obia.classification import cross_validation_splits

outer = split_objects(labelled, method="grouped", groups="site_id",
                      test_size=0.2, random_state=42)
development = labelled.iloc[outer.train_indices].copy()
folds = cross_validation_splits(
    development, method="grouped", groups="site_id", n_splits=3,
)
search = GridSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    {"logisticregression__C": [0.1, 1, 10]},
    cv=[fold.as_sklearn() for fold in folds],
    scoring="balanced_accuracy",
)
result = classify(
    objects, labelled, method=search, split=outer,
    feature_columns=["b0_mean", "b1_mean", "b2_mean"], compute_reports=True,
)
```

The inner indices refer to `development`, whose row order matches the outer
training subset passed to `GridSearchCV`. Alternatively, pass each fold as
`split=fold` to `classify(objects, development, ...)` for manual comparisons.
Use `method="spatial"` with `block_size` and optional buffers in both helpers for
spatial folds. Reuse the same folds across models and inspect their class counts.

Keep scaling, imputation, feature selection, and other learned preprocessing
inside the estimator pipeline so each inner fold fits its own transformations
([scikit-learn guidance](https://scikit-learn.org/stable/common_pitfalls.html)).
OBIA does not replace internal CV used by estimators such as stacking, calibration,
or parameter searches: configure those folds explicitly when their defaults
would split your groups. Tune segmentation or feature settings using development
data too, without repeatedly consulting the final test labels.

After selecting settings and reporting the final held-out result, a fresh call
with `test_size=None` can fit all labelled objects. That refitted model no longer
has an untouched test subset in the same table; the earlier report describes the
model evaluated before refitting.

## Distinguish spatial transfer from map accuracy

Spatial validation estimates performance under a specified separation from
training data. It is not automatically an unbiased estimate of a finished map's
accuracy across its entire area. Holding out distant blocks can overstate error
when much of the mapped area is close to training observations
([Wadoux et al., 2021](https://doi.org/10.1016/j.ecolmodel.2021.109692)).

For a defensible map accuracy or land-cover area assessment, plan an independent
reference sample with a probability sampling design, appropriate inclusion
weights, and uncertainty estimates
([Olofsson et al., 2014](https://doi.org/10.1016/j.rse.2014.02.015)).
OBIA's classifier report is an **unweighted object-level** report: large and small
segments each count once. It does not implement a survey-weighted or area-adjusted
accuracy assessment. Choosing `method="explicit"` preserves a prescribed sample
partition, but does not by itself perform that statistical analysis.

## References

- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal,
  spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929.
  [doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881).
- Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area
  of applicability of spatial prediction models. *Methods in Ecology and Evolution*,
  12, 1620–1633. [doi:10.1111/2041-210X.13650](https://doi.org/10.1111/2041-210X.13650).
- Olofsson, P., et al. (2014). Good practices for estimating area and assessing
  accuracy of land change. *Remote Sensing of Environment*, 148, 42–57.
  [doi:10.1016/j.rse.2014.02.015](https://doi.org/10.1016/j.rse.2014.02.015).
- Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S., & Brus, D. J. (2021).
  Spatial cross-validation is not the right way to evaluate map accuracy.
  *Ecological Modelling*, 457, 109692.
  [doi:10.1016/j.ecolmodel.2021.109692](https://doi.org/10.1016/j.ecolmodel.2021.109692).
