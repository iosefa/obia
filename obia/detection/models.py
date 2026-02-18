"""
RetinaNet-based Detection Model (Modified for N-Channel Input)
-------------------------------------------------------------

Allows specifying 'in_channels' for multi-band data. By default, the
pretrained backbone is for 3 channels. We replace the first conv layer
to match in_channels, partially or fully copying weights for the
first 3 channels if in_channels >= 3.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import (
    retinanet_resnet50_fpn
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def build_detection_model(num_classes=2, in_channels=3):
    """
    Builds a RetinaNet model with optional adjustment for N-band input.

    Args:
        num_classes (int): Number of classes (including background if you prefer).
        in_channels (int): Number of input channels (e.g., 4 for RGB+CHM).

    Returns:
        model (nn.Module): The modified RetinaNet model.
    """
    model = retinanet_resnet50_fpn(weights=None)
    model.transform.image_mean = [0.0] * 10
    model.transform.image_std = [1.0] * 10
    model.train()

    anchor_generator = model.anchor_generator
    out_channels = model.backbone.out_channels
    num_anchors = anchor_generator.num_anchors_per_location()[0]

    model.head.classification_head = RetinaNetClassificationHead(
        out_channels,
        num_anchors,
        num_classes
    )

    if in_channels != 3:
        old_conv = model.backbone.body.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            n_to_copy = min(in_channels, 3)
            new_conv.weight[:, :n_to_copy, :, :] = old_conv.weight[:, :n_to_copy, :, :]

        model.backbone.body.conv1 = new_conv

    return model
