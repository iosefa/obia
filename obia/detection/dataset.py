"""
Tree Detection Dataset
----------------------

This module contains a custom PyTorch Dataset implementation for tree detection tasks.
"""
import os
import json
import cv2
import torch
from torch.utils.data import Dataset


class TreeDetectionDataset(Dataset):
    """
    A custom Dataset for object detection (tree crowns, etc.).
    Expects annotation structure like:
        {
          "image_1": {
            "file_name": "image_1.jpg",
            "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
            "labels": [1, 1, ...]
          },
          "image_2": {
            ...
          }
        }
    """

    def __init__(self, images_dir, annotations_path, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)

        self.image_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann = self.annotations[image_id]

        # Load and convert the image
        image_path = os.path.join(self.images_dir, ann["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # IMPORTANT: Keep boxes/labels as lists for Albumentations
        boxes = ann["boxes"]   # e.g. [[x1, y1, x2, y2], ...]
        labels = ann["labels"] # e.g. [1, 1, ...]

        # Apply Albumentations transforms if provided
        if self.transforms is not None:
            augmented = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

        # Convert to tensors AFTER augmentation
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return image, target