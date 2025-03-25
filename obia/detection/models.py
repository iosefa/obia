"""
RetinaNet-based Detection Model
-------------------------------

This module provides a custom object detection model built on the RetinaNet architecture.
"""
import torch
from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def build_detection_model(num_classes=2):
    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    model.train()

    anchor_generator = model.anchor_generator
    out_channels = model.backbone.out_channels
    num_anchors = anchor_generator.num_anchors_per_location()[0]

    model.head.classification_head = RetinaNetClassificationHead(
        out_channels,
        num_anchors,
        num_classes
    )
    return model