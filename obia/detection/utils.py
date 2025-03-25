import cv2
import matplotlib.pyplot as plt
import torch

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    Resize, Normalize
)
from albumentations.pytorch import ToTensorV2


def get_transforms(train=True):
    """
    Returns data augmentation and preprocessing transforms.
    For detection tasks, must configure 'bbox_params'.
    """
    if train:
        return Compose(
            [
                Resize(512, 512),  # or 224x224, but bigger might help with detection
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params={
                "format": "pascal_voc",
                "label_fields": ["labels"],
                "min_area": 0,
                "min_visibility": 0.0,
            },
        )
    else:
        return Compose(
            [
                Resize(512, 512),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params={
                "format": "pascal_voc",
                "label_fields": ["labels"],
                "min_area": 0,
                "min_visibility": 0.0,
            },
        )


def collate_fn(batch):
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return images, targets


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou


def visualize_predictions(image_path, detection_output):
    image = cv2.imread(image_path)
    for box, score, label in zip(
        detection_output["boxes"],
        detection_output["scores"],
        detection_output["labels"]
    ):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Class {label}, {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()