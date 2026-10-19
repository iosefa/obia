"""Object classification with scikit-learn and optional XGBoost estimators."""

from obia.classification.classify import ClassifiedImage, classify
from obia.classification.estimators import CLASSIFICATION_METHODS, create_classifier

__all__ = ["CLASSIFICATION_METHODS", "ClassifiedImage", "classify", "create_classifier"]
