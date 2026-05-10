import importlib


def test_top_level_package_imports():
    import obia

    assert obia is not None


def test_core_modules_import_without_detection_extras():
    modules = [
        "obia.classification.classify",
        "obia.handlers.geotif",
        "obia.segmentation.segment",
        "obia.segmentation.segment_boundaries",
        "obia.segmentation.segment_statistics",
        "obia.utils.tiling",
    ]

    for module in modules:
        importlib.import_module(module)


def test_detection_package_import_is_lazy():
    module = importlib.import_module("obia.detection")

    assert "build_detection_model" in module.__all__
