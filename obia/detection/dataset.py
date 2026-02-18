import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import rasterio


class TreeDetectionDataset(Dataset):
    """
    Represents a dataset for tree detection tasks.

    This class handles loading, preprocessing, and transforming tree detection
    datasets. Images and annotations are loaded and preprocessed for deep learning
    models. It supports geometric and color augmentations if transforms are provided,
    and optional scaling of pixel values.

    :ivar images_dir: Path to the directory containing image files.
    :type images_dir: str
    :ivar annotations: Parsed annotations for the dataset, loaded from the JSON file.
    :type annotations: dict
    :ivar image_ids: List of image IDs corresponding to the keys in the annotations.
    :type image_ids: list
    :ivar transforms: A callable for data augmentation and transformations. It must support
        the `image`, `bboxes`, and `labels` keys for input and output.
    :type transforms: callable, optional
    :ivar do_scale: Whether to scale image pixel values to the range 0-255.
    :type do_scale: bool
    """
    def __init__(self, images_dir, annotations_path, transforms=None, do_scale=True):
        self.images_dir = images_dir
        self.transforms = transforms
        self.do_scale = do_scale

        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)
        self.image_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann = self.annotations[image_id]
        image_path = os.path.join(self.images_dir, ann["file_name"])

        with rasterio.open(image_path) as src:
            image_array = src.read()

        image_array = np.transpose(image_array, (1, 2, 0))

        if self.do_scale:
            data_min = image_array.min()
            data_max = image_array.max()
            if data_max > data_min:
                image_array = 255.0 * (image_array - data_min) / (data_max - data_min + 1e-8)
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        boxes = ann["boxes"]
        labels = ann["labels"]

        if self.transforms is not None:
            augmented = self.transforms(
                image=image_array,
                bboxes=boxes,
                labels=labels
            )
            image_array = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

        image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return image_tensor, target
