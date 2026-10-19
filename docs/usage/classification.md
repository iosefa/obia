# Classification

Classification learns from labelled objects and predicts a class for each object
in your map. Start with polygons containing numeric feature columns, then choose
a classifier and the objects to use for training and evaluation.

## Prepare training labels

Load the object table produced by [segmentation](segmentation.md) or
[feature extraction](raster-features.md). Labelled points need a `class` column
and coordinates in the same CRS as the objects:

```python
import geopandas as gpd

from obia.utils.utils import label_segments

segments_gdf = gpd.read_file("segments.gpkg")
labelled_points = gpd.read_file("labelled_points.gpkg").to_crs(segments_gdf.crs)
training_segments, mixed_segments = label_segments(segments_gdf, labelled_points)
```

`training_segments` has a `feature_class` column. Labels can be strings such as
`"forest"` or numeric class codes such as `10`, `30`, and `80`. Remove unlabelled
rows before classification. `mixed_segments` lists IDs that intersect multiple
classes; review these objects before training.

Calculate the features you need before labelling, or recalculate them on both
training and prediction polygons with the same settings. See
[raster object features](raster-features.md).

## Train and predict

This example uses band means to fit a random forest. The default split reserves
20% of labelled objects for a random holdout. For evaluation in new fields or
regions, [choose a grouped or spatial split](validation.md) and pass it as `split`.

```python
from obia.classification import classify

result = classify(
    segments_gdf,
    training_segments,
    method="rf",
    feature_columns=["b0_mean", "b1_mean", "b2_mean"],
    n_estimators=300,
    random_state=42,
    compute_reports=True,
)

classified = result.classified
print(result.report)
```

The result contains:

- `classified`: a copy of the input GeoDataFrame with `predicted_class` and
  `prediction_margin`; geometry, CRS, row indices, and other columns are retained.
- `model`: the fitted scikit-learn pipeline, including any OBIA scaler.
- `feature_columns`: the ordered feature names used during training.
- `classes`: labels in probability-column and confusion-matrix order.
- `probabilities`: a DataFrame with one column per class, or `None` for classifiers
  without `predict_proba`.
- `confusion_matrix` and `report`: held-out metrics when requested.
- `params`: the classifier's parameters.
- `split`: the reusable partition, with train/test/excluded row positions,
  map-ready assignments, and class counts.

If `feature_columns` is omitted, OBIA selects numeric training columns except
`segment_id`, `feature_class`, `predicted_class`, and `prediction_margin`. Text
metadata and geometry are excluded. Prediction columns are selected in the same
order, regardless of the segment table's column order. Missing required columns
raise an error; extra prediction columns are ignored. Use an explicit list when
numeric metadata should not be used as a feature.

Selected feature dtypes are preserved, including integer and nullable numeric
columns, so dtype-based selectors in custom pipelines work as expected.

## Training and evaluation

`scale_features="auto"` applies `StandardScaler` to the named linear, SVM,
neighbor, MLP, discriminant-analysis, and Gaussian-process classifiers, including
instances of those same classes. Tree models, Naive Bayes, XGBoost, custom
pipelines, and composite estimators receive features without an added scaler.
Pipelines and composite estimators should include their own preprocessing,
especially when it needs to be fitted inside cross-validation folds.
`scale_features=True` or `False` overrides the automatic choice.

OBIA fits preprocessing only on the training partition and reuses it for test and
prediction data, following
[scikit-learn's preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html).

By default, `test_size=0.2` holds out 20% of labelled objects, with a stratified
split and `split_random_state=42`. The classifier fits the remaining 80%, whether
or not reports are requested. `random_state` in additional keyword arguments
controls the estimator separately. Set `compute_reports=True` for a confusion
matrix and text classification report; their class order is `result.classes`.

Each class needs enough examples for the requested stratified split. Adjust
`test_size` or collect more labels when a split cannot represent every class.
`stratify=False` requests an ordinary random split, but all classes must still
occur in its training partition. Set `test_size=None` to fit all labelled objects;
held-out reports are unavailable in that mode. OBIA does not refit automatically
on the full labelled table after evaluation.

For predictions in new sites or areas, create a grouped or spatial split before
fitting. Reuse it across models and inspect class coverage and exclusions:

```python
from obia.classification import split_objects

partition = split_objects(training_segments, method="grouped", groups="site_id",
                          test_size=0.2, random_state=42)
result = classify(segments_gdf, training_segments, method="rf", split=partition,
                  compute_reports=True, random_state=42)
print(result.split.class_counts)
```

This example requires a `site_id` column identifying independent sampling units.
A named grouping column is excluded from predictors. Supplying `split` overrides
the random split options on `classify()`. See
[training and validation](validation.md) for spatial blocks,
geometry buffers, explicit test sets, cross-validation, and the distinction
between spatial transfer and map accuracy assessment, with literature references.

## Choose a classifier

Pass a method name and the estimator's constructor parameters to `classify()`.
Class names in the right column also work as method strings, ignoring case.

| Method | Estimator |
| --- | --- |
| `rf` | `RandomForestClassifier` |
| `extra_trees` | `ExtraTreesClassifier` |
| `decision_tree` | `DecisionTreeClassifier` |
| `extra_tree` | `ExtraTreeClassifier` |
| `gradient_boosting` | `GradientBoostingClassifier` |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier` |
| `adaboost` | `AdaBoostClassifier` |
| `bagging` | `BaggingClassifier` |
| `mlp` | `MLPClassifier` |
| `logistic_regression` | `LogisticRegression` |
| `logistic_regression_cv` | `LogisticRegressionCV` |
| `ridge` | `RidgeClassifier` |
| `ridge_cv` | `RidgeClassifierCV` |
| `sgd` | `SGDClassifier` |
| `perceptron` | `Perceptron` |
| `passive_aggressive` | `PassiveAggressiveClassifier` |
| `svc` | `SVC` |
| `nu_svc` | `NuSVC` |
| `linear_svc` | `LinearSVC` |
| `knn` | `KNeighborsClassifier` |
| `radius_neighbors` | `RadiusNeighborsClassifier` |
| `nearest_centroid` | `NearestCentroid` |
| `gaussian_nb` | `GaussianNB` |
| `bernoulli_nb` | `BernoulliNB` |
| `multinomial_nb` | `MultinomialNB` |
| `complement_nb` | `ComplementNB` |
| `categorical_nb` | `CategoricalNB` |
| `lda` | `LinearDiscriminantAnalysis` |
| `qda` | `QuadraticDiscriminantAnalysis` |
| `gaussian_process` | `GaussianProcessClassifier` |
| `dummy` | `DummyClassifier` |
| `voting` | `VotingClassifier` |
| `stacking` | `StackingClassifier` |
| `calibrated` | `CalibratedClassifierCV` |
| `one_vs_rest` | `OneVsRestClassifier` |
| `one_vs_one` | `OneVsOneClassifier` |
| `output_code` | `OutputCodeClassifier` |
| `xgboost` | `XGBClassifier` |
| `xgboost_rf` | `XGBRFClassifier` |

Aliases include `random_forest`, `svm`, `xgb`, and `xgbrf`. To list method names or
create an unfitted estimator:

```python
from obia.classification import CLASSIFICATION_METHODS, create_classifier

print(CLASSIFICATION_METHODS)
estimator = create_classifier("extra_trees", n_estimators=300, random_state=42)
```

The named classifiers use their upstream defaults. Their data requirements still
apply: multinomial/complement Naive Bayes need nonnegative features, categorical
Naive Bayes needs nonnegative integer categories, and radius neighbors needs an
appropriate radius or outlier policy. `passive_aggressive` follows the upstream
class, which is deprecated in newer scikit-learn releases. See the
[scikit-learn supervised learning guide](https://scikit-learn.org/stable/supervised_learning.html)
for estimator parameters and assumptions.

### Support vector machines, neighbors, and neural networks

```python
svm_result = classify(segments_gdf, training_segments, method="svc", C=10, kernel="rbf")
knn_result = classify(segments_gdf, training_segments, method="knn", n_neighbors=7)
mlp_result = classify(
    segments_gdf, training_segments, method="mlp",
    hidden_layer_sizes=(100, 50), max_iter=500, random_state=42,
)
```

SVC and LinearSVC work without probability estimates. In that case,
`prediction_margin` is `NaN` and `probabilities` is `None`. To add probabilities,
use a calibrated estimator as shown below.

### XGBoost

XGBoost is included in the standard installation:

```python
xgb_result = classify(
    segments_gdf, training_segments,
    method="xgboost",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    n_jobs=4,
    random_state=42,
)
```

Use `method="xgboost_rf"` for XGBoost's random-forest classifier. OBIA encodes class
labels internally and restores the original strings or codes in predictions,
probability columns, and reports. XGBoost is imported only when requested.
Constructor options follow the
[XGBoost scikit-learn interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html).
The `XGBRFClassifier` wrapper is deprecated in newer XGBoost releases.
OBIA does not supply an `eval_set`; manage early stopping through a separately
configured training workflow.

### Custom estimators and pipelines

Pass any scikit-learn-compatible classifier instance as `method`. OBIA clones it,
so your supplied object remains unchanged. Pipelines support feature imputation,
selection, transformations, calibrated classifiers, and other installed
scikit-learn classifiers. The target remains one `feature_class` per object;
multioutput/multilabel targets are outside this API.

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

pipeline_result = classify(
    segments_gdf, training_segments,
    method=pipeline,
    classifier__C=0.5,
    test_size=None,
)
```

Keyword arguments use `set_params` for supplied estimators, including nested
names such as `classifier__C`. Feature selectors and searches such as
`GridSearchCV` use the estimator's usual parameter names. Missing feature values are retained for
native estimator support or an imputer pipeline. OBIA rejects infinities and
does not silently fill missing values. Estimators without missing-value support
raise their normal scikit-learn error.

Predictors must be real numeric columns. Encode categorical or string predictors
before calling `classify()`. A custom estimator must follow scikit-learn's
classifier interface, including fitted `classes_` and one prediction per row.

Raw precomputed kernels or distance matrices are outside this input contract and
raise an explicit error. A custom pipeline may build its own Gram and cross-kernel
matrices from the object features; that keeps matrix construction inside each
training split. See the [SVC input requirements](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

Composite named methods require their usual estimator arguments:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

calibrated_svm = CalibratedClassifierCV(
    Pipeline([("scale", StandardScaler()), ("svc", LinearSVC())]), cv=3,
)
calibrated_result = classify(segments_gdf, training_segments, method=calibrated_svm)

voting_result = classify(
    segments_gdf, training_segments, method="voting", voting="soft",
    estimators=[
        ("tree", DecisionTreeClassifier(random_state=42)),
        ("linear", pipeline),
    ],
)
```

### Use a fitted model

On scikit-learn versions that provide `FrozenEstimator`, a supplied frozen model
retains its fitted state. OBIA does not refit it or add a label encoder inside it;
its class labels must already match your objects' labels. Evaluate a frozen model
only on objects excluded from its original training data.

## Probabilities and spatial class constraints

`prediction_margin` is the difference between the two largest class
probabilities. It is a probability gap, not a calibrated accuracy estimate.
Models without probabilities retain their normal `predict` result and get a
`NaN` margin. For SVC configurations that expose probabilities, the most probable
class can differ from `predict`; OBIA preserves the estimator's prediction when
there are no spatial restrictions.

An optional `acceptable_classes_gdf` limits classes within polygons. In this
example, `study_area_polygon` is a Shapely polygon defining the restricted area:

```python
allowed = gpd.GeoDataFrame(
    {"acceptable_classes": [["forest", "soil"]]},
    geometry=[study_area_polygon],
    crs=segments_gdf.crs,
)
constrained = classify(
    segments_gdf, training_segments,
    method="rf", acceptable_classes_gdf=allowed,
)
```

Constraint polygons must use the segments' CRS and their lists must contain the
original class labels. For each intersecting segment, OBIA chooses the largest
probability among permitted learned classes. Overlapping polygons intersect
their allowed-class sets. An unmatched segment is unrestricted; no permitted
learned classes produces a missing prediction. A single permitted class produces
that class and a `NaN` margin. With two or more permitted classes, the margin uses
the original probabilities without renormalizing. `result.probabilities` always
contains the original, unrestricted probabilities.

Spatial restrictions require `predict_proba`; use `CalibratedClassifierCV` for a
classifier that lacks it. The returned model represents the statistical
classifier; calling `result.model.predict()` does not apply polygon restrictions.

## Explain predictions with SHAP

The `explain` extra requires SHAP 0.51 or newer for compatibility with the supported
NumPy and XGBoost versions.

```bash
pip install "obia[explain]"
```

```python
explained = classify(
    segments_gdf, training_segments, method="rf",
    n_estimators=300, compute_shap=True, sample_shap=True,
)
```

Random forests, extra trees, decision trees, and XGBoost tree models use
`TreeExplainer` when they are the final classifier in OBIA's pipeline. Other
probabilistic estimators and supplied pipelines use model-agnostic
`KernelExplainer`, which can be expensive. Models without probabilities require
calibration for this path. `sample_shap=True` limits both the background and the
explained training rows to at most 500. Explaining a model does not change the
rows used to fit it. Values are returned in `result.shap_values`; tree explanations
use the classifier's input feature space, including scaling when enabled.

## Save predictions

```python
result.classified.to_file("classified_segments.gpkg", driver="GPKG")
```

The output is a vector object map. To create a raster map, rasterize the predicted
polygons onto the source grid. The GeoPackage does not store the fitted model or
its preprocessing pipeline.
