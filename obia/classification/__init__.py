"""Object and pixel classification with scikit-learn and XGBoost estimators."""

from obia.classification.classify import ClassifiedImage, classify
from obia.classification.estimators import CLASSIFICATION_METHODS, create_classifier
from obia.classification.pixels import ClassifiedRaster, classify_pixels, sample_pixels
from obia.classification.splitting import ObjectSplit, cross_validation_splits, split_objects

__all__ = ["CLASSIFICATION_METHODS", "ClassifiedImage", "ClassifiedRaster", "classify",
           "classify_pixels", "sample_pixels", "create_classifier",
           "ObjectSplit", "split_objects", "cross_validation_splits"]
