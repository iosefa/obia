"""Small, georeferenced datasets included with OBIA for examples."""

import json
from pathlib import Path


def chesapeake_sample(split: str = "train") -> dict:
    """Locate a bundled Chesapeake imagery/reference pair without a download.

    Args:
        split: ``"train"`` or ``"test"``. These are 1024 × 1024 pixel crops
            from separate tiles in the publisher's Maryland train/test sets.

    Returns:
        dict: ``image`` and ``reference`` are local GeoTIFF Paths;
            ``class_names`` maps integer reference codes to names; ``metadata``
            records the source tile, crop window, grid, URLs, and checksums.
            The imagery has four bands in original order, selected in OBIA as
            indices 0–3. The reference uses codes 1–6, with 15 reserved for NoData.

    The reference is a land-cover map, not independent field observations.
    See the accompanying ``chesapeake/README.md`` for attribution and terms.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")
    directory = Path(__file__).parent / "chesapeake"
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    sample = metadata["samples"][split]
    return {
        "image": directory / sample["files"]["image"]["filename"],
        "reference": directory / sample["files"]["reference"]["filename"],
        "class_names": {int(code): name for code, name in metadata["class_names"].items()},
        "metadata": sample,
    }
