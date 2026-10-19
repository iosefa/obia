"""Named classifiers and support for scikit-learn-compatible estimators."""

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier,
    Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier,
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


_CLASSIFIERS = {
    "rf": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "decision_tree": DecisionTreeClassifier,
    "extra_tree": ExtraTreeClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "adaboost": AdaBoostClassifier,
    "bagging": BaggingClassifier,
    "mlp": MLPClassifier,
    "logistic_regression": LogisticRegression,
    "logistic_regression_cv": LogisticRegressionCV,
    "ridge": RidgeClassifier,
    "ridge_cv": RidgeClassifierCV,
    "sgd": SGDClassifier,
    "perceptron": Perceptron,
    "passive_aggressive": PassiveAggressiveClassifier,
    "svc": SVC,
    "nu_svc": NuSVC,
    "linear_svc": LinearSVC,
    "knn": KNeighborsClassifier,
    "radius_neighbors": RadiusNeighborsClassifier,
    "nearest_centroid": NearestCentroid,
    "gaussian_nb": GaussianNB,
    "bernoulli_nb": BernoulliNB,
    "multinomial_nb": MultinomialNB,
    "complement_nb": ComplementNB,
    "categorical_nb": CategoricalNB,
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
    "gaussian_process": GaussianProcessClassifier,
    "dummy": DummyClassifier,
    "voting": VotingClassifier,
    "stacking": StackingClassifier,
    "calibrated": CalibratedClassifierCV,
    "one_vs_rest": OneVsRestClassifier,
    "one_vs_one": OneVsOneClassifier,
    "output_code": OutputCodeClassifier,
}
_XGBOOST_METHODS = {"xgboost": "XGBClassifier", "xgboost_rf": "XGBRFClassifier"}
CLASSIFICATION_METHODS = tuple(_CLASSIFIERS) + tuple(_XGBOOST_METHODS)
"""Method names accepted by object and pixel classification and create_classifier."""

_ALIASES = {cls.__name__.lower(): name for name, cls in _CLASSIFIERS.items()}
_ALIASES.update({cls.lower(): name for name, cls in _XGBOOST_METHODS.items()})
_ALIASES.update(random_forest="rf", svm="svc", xgb="xgboost", xgbrf="xgboost_rf")
_SCALED_TYPES = (
    MLPClassifier, LogisticRegression, LogisticRegressionCV, RidgeClassifier,
    RidgeClassifierCV, SGDClassifier, Perceptron, PassiveAggressiveClassifier,
    SVC, NuSVC, LinearSVC, KNeighborsClassifier, RadiusNeighborsClassifier,
    NearestCentroid, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
    GaussianProcessClassifier,
)


def create_classifier(method="rf", **kwargs):
    """Create an unfitted classifier, or clone a supplied estimator or pipeline.

    ``method`` accepts a name in ``CLASSIFICATION_METHODS``, a scikit-learn
    class name (for example ``"SVC"``), or a classifier instance. Keyword
    arguments are constructor parameters for named methods and ``set_params``
    overrides for instances. XGBoost ships with OBIA and is imported only when
    requested. Composite classifiers such as
    voting and stacking require their usual ``estimators`` argument.
    """
    if isinstance(method, str):
        name = method.lower().replace("-", "_")
        name = _ALIASES.get(name, name)
        if name in _XGBOOST_METHODS:
            import xgboost
            estimator = getattr(xgboost, _XGBOOST_METHODS[name])(**kwargs)
        elif name in _CLASSIFIERS:
            estimator = _CLASSIFIERS[name](**kwargs)
        else:
            raise ValueError(
                f"Unknown classification method {method!r}. Choose from "
                f"{', '.join(CLASSIFICATION_METHODS)}, or pass a classifier instance."
            )
    else:
        if (isinstance(method, type)
                or not all(callable(getattr(method, attr, None)) for attr in ("fit", "predict", "get_params"))
                or not is_classifier(method)):
            raise TypeError("method must be a classifier name or a scikit-learn classifier instance.")
        estimator = clone(method)
        # FrozenEstimator intentionally returns itself from clone. Parameter
        # overrides must not mutate it, including frozen steps inside pipelines.
        if estimator is method or kwargs:
            estimator = deepcopy(estimator)
        estimator.set_params(**kwargs)
        return estimator
    # Also clone estimator-valued parameters of named composite models.
    return clone(estimator)


def _scale_automatically(estimator):
    # Pipelines and composite/custom estimators own their preprocessing.
    return isinstance(estimator, _SCALED_TYPES)


def _requires_pairwise_input(estimator):
    # Public estimator tags were introduced after our minimum scikit-learn version.
    if hasattr(BaseEstimator, "__sklearn_tags__"):
        return estimator.__sklearn_tags__().input_tags.pairwise
    return getattr(estimator, "_get_tags", lambda: {})().get("pairwise", False)


def _classifier_pipeline(method, scale_features, **kwargs):
    """Use identical estimator adaptation and preprocessing in both workflows."""
    estimator = create_classifier(method, **kwargs)
    if _requires_pairwise_input(estimator):
        raise ValueError(
            "Precomputed kernels/distances are not feature columns. Pass ordinary "
            "features, or a pipeline that builds its pairwise matrices from those features."
        )
    params = estimator.get_params()
    if scale_features != "auto" and not isinstance(scale_features, (bool, np.bool_)):
        raise ValueError("scale_features must be 'auto', True, or False.")
    scale = _scale_automatically(estimator) if scale_features == "auto" else scale_features
    steps = [("scaler", StandardScaler())] if scale else []
    return Pipeline(steps + [("classifier", _encode_xgboost(estimator))]), params


def _is_xgboost_classifier(estimator):
    # Inspect the MRO so subclasses work without eagerly importing XGBoost.
    return any(base.__module__.split(".")[0] == "xgboost" and base.__name__ == "XGBClassifier"
               for base in type(estimator).__mro__)


class _EncodedClassifier(ClassifierMixin, BaseEstimator):
    """Give XGBoost ordinary class labels while keeping its internal labels 0..K-1."""

    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep=True):
        parameters = super().get_params(deep=deep)
        if deep:
            # Keep existing GridSearchCV/Pipeline parameter paths valid after wrapping.
            parameters.update(self.estimator.get_params(deep=True))
        return parameters

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        self.estimator.set_params(**{key.removeprefix("estimator__"): value
                                     for key, value in params.items()})
        return self

    def __sklearn_tags__(self):
        tags = deepcopy(self.estimator.__sklearn_tags__())
        tags.target_tags.multi_output = False
        tags.classifier_tags.multi_label = False
        return tags

    def _more_tags(self):
        # Compatibility with scikit-learn 1.4/1.5, including native NaN support.
        tags = self.estimator._get_tags().copy()
        tags.update(multioutput=False, multilabel=False)
        return tags

    def fit(self, X, y, sample_weight=None):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self.encoder_.transform(y), sample_weight=sample_weight)
        return self

    @property
    def n_features_in_(self):
        return self.estimator_.n_features_in_

    @property
    def feature_names_in_(self):
        return self.estimator_.feature_names_in_

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_

    @property
    def coef_(self):
        return self.estimator_.coef_

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X).astype(int))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def _encode_xgboost(estimator):
    # A frozen estimator is already fitted; adding an unfitted label adapter
    # would break prediction and can modify a shared step retained by clone.
    if any(base.__module__.startswith("sklearn.frozen") and base.__name__ == "FrozenEstimator"
           for base in type(estimator).__mro__):
        return estimator
    if isinstance(estimator, _EncodedClassifier):
        return estimator
    if _is_xgboost_classifier(estimator):
        return _EncodedClassifier(estimator)
    if isinstance(estimator, dict):
        return {name: _encode_xgboost(value) for name, value in estimator.items()}
    if isinstance(estimator, list):
        return [_encode_xgboost(value) for value in estimator]
    if isinstance(estimator, tuple):
        return tuple(_encode_xgboost(value) for value in estimator)
    if isinstance(estimator, BaseEstimator):
        # Include estimator choices in parameter grids and ColumnTransformer tuples.
        estimator.set_params(**{name: _encode_xgboost(value)
                                for name, value in estimator.get_params(deep=False).items()})
    return estimator
