# Experimental Detection

Use `obia.detection` to train a RetinaNet model that locates objects with bounding
boxes. This is an experimental workflow, separate from segmenting and classifying
polygon objects; its interfaces may change.

## Install detection dependencies

Install the detection extra:

```bash
pip install "obia[detection]"
```

The detection extra includes PyTorch, Torchvision, Albumentations, and Matplotlib.

## Prepare image chips and annotations

`TreeDetectionDataset` reads raster chips and JSON annotations:

```python
from obia.detection.dataset import TreeDetectionDataset
dataset = TreeDetectionDataset(
    images_dir="/path/to/chips",
    annotations_path="/path/to/annotations.json",
)
```

The JSON file maps image IDs to filenames, boxes, and class labels. Box
coordinates are `[xmin, ymin, xmax, ymax]` in pixels. For example:

```json
{
  "chip_01": {
    "file_name": "chip_01.tif",
    "boxes": [[20, 30, 80, 95]],
    "labels": [1]
  }
}
```

## Build a model

```python
from obia.detection.models import build_detection_model

model = build_detection_model(
    num_classes=2,
    in_channels=3,
)
```

Set `in_channels` to match the number of raster bands used for detection.

## Train

```python
from torch.utils.data import DataLoader

from obia.detection.train import train_model
from obia.detection.utils import collate_fn

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

model = train_model(
    model,
    loader,
    num_epochs=20,
    device="cpu",
)
```

Use `device="cuda"` for a supported NVIDIA GPU, or `device="mps"` for a supported
Apple Silicon installation. The CPU example works without GPU support.

## Predict

```python
from obia.detection.predict import predict

detections = predict(
    model,
    image_path="/path/to/image.tif",
    device="cpu",
    score_threshold=0.5,
)
```

The result contains `boxes`, `scores`, and `labels` arrays. Boxes use pixel
coordinates; convert them with the source raster's transform if you need map
coordinates.
