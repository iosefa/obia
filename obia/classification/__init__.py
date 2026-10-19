"""Object classification with scikit-learn and XGBoost estimators."""

from obia.classification.classify import ClassifiedImage, classify
from obia.classification.estimators import CLASSIFICATION_METHODS, create_classifier
from obia.classification.splitting import ObjectSplit, cross_validation_splits, split_objects

__all__ = ["CLASSIFICATION_METHODS", "ClassifiedImage", "classify", "create_classifier",
           "ObjectSplit", "split_objects", "cross_validation_splits"]
