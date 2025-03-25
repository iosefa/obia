"""
Object Detection Prediction Script
----------------------------------

This module provides a function for running inference with a custom object detection model.
"""
import torch
import cv2
import numpy as np

def predict(model, image_path, device="cpu", score_threshold=0.5):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model([image_tensor.to(device)])  # list of dict

    boxes = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels
    }