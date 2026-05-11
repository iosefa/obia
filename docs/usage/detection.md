# Experimental Detection

`obia.detection` is experimental. Its interfaces and workflow may change in future releases.

The package provides a RetinaNet-based object detection path for raster imagery. It is separate from the segmentation/classification workflow.

Use detection when the target output is bounding boxes. Use segmentation when the target output is image objects and object-level feature tables.

## Install Experimental Detection Dependencies

Install the detection extra:

```bash
pip install "obia[detection]"
```

The detection extra includes PyTorch, Torchvision, Albumentations, and Matplotlib.

## Dataset

`TreeDetectionDataset` reads raster chips and JSON annotations:

```python
from obia.detection.dataset import TreeDetectionDataset
from obia.detection.utils import get_transforms

dataset = TreeDetectionDataset(
    image_dir="/path/to/chips",
    annotations_file="/path/to/annotations.json",
    transforms=get_transforms(train=True),
)
```

Annotations should provide bounding boxes and labels for each image.

## Build A Model

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
    device="cuda",
)
```

Use `device="mps"` on Apple Silicon when the local PyTorch installation supports it.

## Predict

```python
from obia.detection.predict import predict

detections = predict(
    model,
    image_path="/path/to/image.tif",
    device="cuda",
    score_threshold=0.5,
)
```

The result is a dictionary with `boxes`, `scores`, and `labels` arrays.

## Current Limits

Detection expects prepared chips and annotations. It is best treated as a model-building workflow for bounding boxes, while segmentation is the starting point for object maps and object-level classification.
