"""
Training Script for a RetinaNet-based Detection Model
-----------------------------------------------------

This module provides functionality to train a custom object detection model built on the RetinaNet architecture.
"""
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs, device="cpu"):
    """
    Trains the RetinaNet detection model.

    Args:
        model (nn.Module): The detection model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data (optional).
        num_epochs (int): Number of epochs to train.
        device (str): Device to use ("cpu", "cuda", or "mps").

    Returns:
        model (nn.Module): Trained model.
    """
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        # Optionally run validation

    return model