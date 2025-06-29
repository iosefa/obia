"""
Object Detection Prediction Script (Multi-band + 0..255 Scaling)
----------------------------------------------------------------

Provides a function `predict()` for running inference with a custom
RetinaNet-based model on an N-band raster. Uses rasterio to read
the data, scales each band to [0..255], then feeds to the model.
"""

import torch
import rasterio
import numpy as np

def predict(model, image_path, device="cpu", score_threshold=0.5):
    """
    Args:
        model (nn.Module): Trained RetinaNet model (with in_channels matching your data).
        image_path (str): Path to the multi-band raster (GeoTIFF, etc.).
        device (str): "cpu", "cuda", or "mps".
        score_threshold (float): Minimum confidence for detection.

    Returns:
        dict with { "boxes": nd.array, "scores": nd.array, "labels": nd.array }
    """
    with rasterio.open(image_path) as src:
        image_array = src.read()

    image_array = np.transpose(image_array, (1, 2, 0))

    data_min = image_array.min()
    data_max = image_array.max()
    if data_max > data_min:
        image_array = 255 * (image_array - data_min) / (data_max - data_min + 1e-8)
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1)

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model([image_tensor.to(device)])

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
