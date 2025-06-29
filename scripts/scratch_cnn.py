import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
import laspy
from shapely.geometry import box

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# -------------------------------
# 1. READ & MATCH TIF / LAS
# -------------------------------
TRAINING_GPKG = "/Users/iosefa/repos/obia/docs/example_data/site_1/crowns_labeled.gpkg"
INDIVIDUAL_CROWNS_DIR = "/Users/iosefa/repos/obia/docs/example_data/site_1/individual_crowns"
INDIVIDUAL_CROWNS_POINTS_DIR = "/Users/iosefa/repos/obia/docs/example_data/site_1/individual_crowns_points"

gdf_training = gpd.read_file(TRAINING_GPKG)

tif_files = sorted(glob.glob(os.path.join(INDIVIDUAL_CROWNS_DIR, "*.tif")))
las_files = sorted(glob.glob(os.path.join(INDIVIDUAL_CROWNS_POINTS_DIR, "*.las")))

tif_dict = {}
for tif_path in tif_files:
    base = os.path.splitext(os.path.basename(tif_path))[0]
    tif_dict[base] = tif_path

las_dict = {}
for las_path in las_files:
    base = os.path.splitext(os.path.basename(las_path))[0]
    las_dict[base] = las_path

common_keys = set(tif_dict.keys()).intersection(las_dict.keys())
training_records = []

for key in common_keys:
    tif_path = tif_dict[key]
    las_path = las_dict[key]

    with rasterio.open(tif_path) as src:
        left, bottom, right, top = src.bounds
    tif_bbox = box(left, bottom, right, top)

    candidate_points = gdf_training[gdf_training.geometry.within(tif_bbox)]
    if len(candidate_points) == 0:
        continue
    label = candidate_points.iloc[0]["feature_class"]  # simplistic approach

    training_records.append((tif_path, las_path, label))

print(f"Found {len(training_records)} crown records.")

# -----------------------------------------------------
# 2. DATASET + DATALOADERS
# -----------------------------------------------------
class_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
NUM_CLASSES   = len(class_mapping)
MAX_POINTS = 2048

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # force consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class CrownDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        tif_path, las_path, raw_label = self.records[idx]
        label = class_mapping[raw_label]

        # --- Load TIF ---
        with rasterio.open(tif_path) as src:
            img = src.read()  # shape (C, H, W)
        img = img.astype(np.float32)

        # If 9-band, pick [4,2,1]
        if img.shape[0] == 9:
            img = img[[4, 2, 1], :, :]

        # Potentially clamp or scale if you have large values
        # e.g., if data is in [0..1000], you might do:
        # img = np.clip(img, 0, 1000) / 1000.

        # transform
        img = np.moveaxis(img, 0, -1)  # (H, W, C)
        img = img_transform(img)       # (3, 224, 224)

        # --- Load LAS ---
        with laspy.open(las_path) as f:
            las = f.read()
            x = las.x
            y = las.y
            z = las.z
        points = np.vstack((x, y, z)).T  # (N, 3)

        n_pts = points.shape[0]
        if n_pts > MAX_POINTS:
            # random downsample
            choice = np.random.choice(n_pts, MAX_POINTS, replace=False)
            points = points[choice, :]
        elif n_pts < MAX_POINTS:
            # replicate to fill up
            shortfall = MAX_POINTS - n_pts
            replicate_idxs = np.random.choice(n_pts, shortfall, replace=True)
            replicate_data = points[replicate_idxs, :]
            points = np.concatenate([points, replicate_data], axis=0)
            # now points.shape is exactly (MAX_POINTS, 3)

        points = torch.from_numpy(points).float()  # shape: (MAX_POINTS, 3)

        return img, points, label

# Build dataset
full_dataset = CrownDataset(training_records)

# Train/val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42))

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

# -----------------------------------------------------
# 3. COMBINED NETWORK
# -----------------------------------------------------
class CombinedNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_resnet = models.resnet18(pretrained=True)
        self.resnet_backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        # shape (B,512,1,1)

        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)

        self.classifier = nn.Linear(512 + 64, NUM_CLASSES)

    def forward(self, img, pts):
        # img: (B, 3, 224, 224)
        x_img = self.resnet_backbone(img)   # (B, 512, 1, 1)
        x_img = x_img.view(x_img.size(0), -1)  # (B,512)

        # pts: (B, MAX_POINTS, 3)
        x_pts = F.relu(self.fc1(pts))
        x_pts = F.relu(self.fc2(x_pts))
        x_pts = F.relu(self.fc3(x_pts))
        x_pts, _ = torch.max(x_pts, dim=1)  # (B,256)
        x_pts = F.relu(self.fc4(x_pts))     # (B,128)
        x_pts = F.relu(self.fc5(x_pts))     # (B,64)

        # Combine
        x_cat = torch.cat([x_img, x_pts], dim=1)  # (B,576)
        logits = self.classifier(x_cat)           # (B,4)
        return logits

# -----------------------------------------------------
# 4. TRAINING LOOP
# -----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

NUM_EPOCHS = 50

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for img, pts, label in loader:
        img = img.to(device)       # (B,3,224,224)
        pts = pts.to(device)       # (B,MAX_POINTS,3)
        label = label.to(device)   # (B,)

        optimizer.zero_grad()
        logits = model(img, pts)   # (B,4)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for img, pts, label in loader:
            img = img.to(device)
            pts = pts.to(device)
            label = label.to(device)

            logits = model(img, pts)
            loss = criterion(logits, label)
            running_loss += loss.item() * img.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


# -----------------------------------------------------
# 5. OPTIONAL: FINAL EVALUATION
# -----------------------------------------------------
from sklearn.metrics import classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for img, pts, label in val_loader:
        img = img.to(device)
        pts = pts.to(device)

        logits = model(img, pts)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(label.numpy().tolist())

print("\nValidation Classification Report (labels = 0..3):")
print(classification_report(all_labels, all_preds, digits=4))
