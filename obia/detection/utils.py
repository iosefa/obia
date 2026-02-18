"""
Transforms, Collate Function, and Utility Helpers
-------------------------------------------------
Provides:
  - get_transforms(): Albumentations pipelines for train/val
  - collate_fn(): custom collate for object detection
  - calculate_iou(): compute IoU between two boxes
  - visualize_predictions(): draw detection boxes and scores on an image
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt


def get_transforms(train=True):
    """
    Returns Albumentations transforms for bounding-box tasks.
    If you have more than 3 channels, consider removing any
    3-channel-specific Normalization or adjusting mean/std.
    """
    if train:
        return A.Compose(
            [
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_area=0,
                min_visibility=0.0
            )
        )
    else:
        return A.Compose(
            [
                # A.Normalize(mean=(...), std=(...)),
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_area=0,
                min_visibility=0.0
            )
        )


def collate_fn(batch):
    """
    Custom collate function for object detection in PyTorch.
    Returns lists of images and targets.
    """
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return images, targets


def calculate_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.
    Boxes assumed in format [x_min, y_min, x_max, y_max].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou


def visualize_predictions(image_path, detection_output, score_threshold=0.0):
    """
    Draws bounding boxes (and scores) on an image and displays it using matplotlib.

    Args:
        image_path (str): Path to the image file.
        detection_output (dict): Must contain "boxes", "scores", "labels".
        score_threshold (float): Only visualize boxes with score >= threshold.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = detection_output["boxes"]
    scores = detection_output["scores"]
    labels = detection_output["labels"]

    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(
                image_rgb,
                (x1, y1),
                (x2, y2),
                (0, 255, 0), 2
            )
            text = f"Class {label}, {score:.2f}"
            cv2.putText(
                image_rgb,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()
