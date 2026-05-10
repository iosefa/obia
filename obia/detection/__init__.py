__all__ = ["build_detection_model", "train_model", "predict", "calculate_iou"]


def __getattr__(name):
    if name == "build_detection_model":
        from .models import build_detection_model

        return build_detection_model
    if name == "train_model":
        from .train import train_model

        return train_model
    if name == "predict":
        from .predict import predict

        return predict
    if name == "calculate_iou":
        from .utils import calculate_iou

        return calculate_iou
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
