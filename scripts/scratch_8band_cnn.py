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
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

####################
# 1. Read training data & match TIF/ LAS (excluding class 6)
####################

TRAINING_GPKG = "/Users/iosefa/trial_small/training.gpkg"
INDIVIDUAL_CROWNS_DIR = "/Users/iosefa/trial_small/individual_crowns"
INDIVIDUAL_CROWNS_POINTS_DIR = "/Users/iosefa/trial_small/individual_crowns_points"

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
    label = candidate_points.iloc[0]["class"]  # simplistic approach

    # Exclude samples with class 6 (the "Other" class)
    if label == 6:
        continue

    training_records.append((tif_path, las_path, label))


####################
# 2. PyTorch Dataset
####################
class CrownDataset(Dataset):
    def __init__(self, records, img_transform=None, max_points=2048):
        self.records = records
        self.img_transform = img_transform
        self.max_points = max_points

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        tif_path, las_path, label = self.records[idx]

        # Load image (multispectral image)
        with rasterio.open(tif_path) as src:
            img = src.read()  # shape: (bands, H, W)
        img = torch.from_numpy(img).float()  # (C, H, W)

        # Use only the first 8 bands (drop any 9th band, e.g. CHM)
        if img.shape[0] > 8:
            img = img[:8, :, :]

        if self.img_transform:
            # Additional transforms can be applied here if needed.
            pass

        # Load LAS (point cloud)
        with laspy.open(las_path) as f:
            las = f.read()
            x = las.x
            y = las.y
            z = las.z
        points = np.vstack((x, y, z)).T  # shape: (N, 3)

        if points.shape[0] > self.max_points:
            choice = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[choice, :]

        points = torch.from_numpy(points).float()  # shape: (N, 3)

        return {
            'image': img,   # shape: (8, H, W)
            'points': points,
            'label': label
        }

img_transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = CrownDataset(training_records, img_transform=img_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


####################
# 3. Define PyTorch Feature Extractors (ResNet for 8-band images + PointNet)
####################
class ResNet8BandFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            feat_size = 512
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            feat_size = 2048
        else:
            raise ValueError("Unsupported model_name")
        # Modify the first conv layer to accept 8 channels instead of 3.
        original_conv = model.conv1  # originally Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        new_conv = nn.Conv2d(
            8,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        with torch.no_grad():
            # Copy the pretrained weights for the first 3 channels...
            old_weight = original_conv.weight  # shape: (64, 3, 7, 7)
            new_weight = torch.zeros((original_conv.out_channels, 8, original_conv.kernel_size[0], original_conv.kernel_size[1]))
            new_weight[:, :3, :, :] = old_weight
            # ...and for channels 4-8, use the average of the original weights.
            avg_weight = old_weight.mean(dim=1, keepdim=True)  # shape: (64, 1, 7, 7)
            new_weight[:, 3:, :, :] = avg_weight.repeat(1, 5, 1, 1)
            new_conv.weight.copy_(new_weight)
        model.conv1 = new_conv
        # Remove final fully connected layer
        model.fc = nn.Identity()
        self.model = model
        self._out_features = feat_size

    def forward(self, x):
        return self.model(x)

    @property
    def output_dim(self):
        return self._out_features


class PointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc4 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc5 = nn.Linear(hidden_dim * 2, hidden_dim)
        self._out_features = hidden_dim

    def forward(self, x):
        # x: (B, N, 3)
        batch_size, num_points, _ = x.shape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Max pooling over points
        x = torch.max(x, dim=1)[0]  # shape: (B, hidden_dim*4)
        x = F.relu(self.fc4(x))     # shape: (B, hidden_dim*2)
        x = F.relu(self.fc5(x))     # shape: (B, hidden_dim)
        return x

    @property
    def output_dim(self):
        return self._out_features


####################
# 4. Inference-Mode Feature Extraction
####################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_feature_extractor = ResNet8BandFeatureExtractor('resnet18', pretrained=True).to(device)
point_feature_extractor = PointNetFeatureExtractor(input_dim=3, hidden_dim=64).to(device)

image_feature_extractor.eval()
point_feature_extractor.eval()

X_features = []
y_labels = []

with torch.no_grad():
    for batch in dataloader:
        # DataLoader returns batched tensors:
        #   img: shape (1, 8, H, W), pts: shape (1, N, 3)
        img = batch['image'].to(device)
        pts = batch['points'].to(device)
        label = batch['label'].item()

        img_feats = image_feature_extractor(img)  # shape: (1, 512)
        pt_feats = point_feature_extractor(pts)     # shape: (1, 64)

        combined_feat = torch.cat([img_feats, pt_feats], dim=1)  # shape: (1, 576)
        combined_feat_np = combined_feat.cpu().numpy().squeeze(axis=0)  # shape: (576,)
        X_features.append(combined_feat_np)
        y_labels.append(label)

X_features = np.array(X_features)  # shape: (num_samples, 576)
y_labels = np.array(y_labels)

print("Feature matrix shape:", X_features.shape)
print("Labels shape:", y_labels.shape)

####################
# 5. Train scikit-learn MLP and apply threshold-based rejection
####################
from sklearn.model_selection import train_test_split

# Our available classes are 1, 2, and 4.
# Remap them to 0, 1, and 2.
class_mapping = {1: 0, 2: 1, 4: 2}
y_mapped = np.array([class_mapping[label] for label in y_labels])

print("Original label distribution:")
for label in sorted(class_mapping.keys()):
    print(f"Class {label}: {np.sum(y_labels == label)}")

print("\nRemapped label distribution:")
for new_label in np.unique(y_mapped):
    print(f"New Class {new_label}: {np.sum(y_mapped == new_label)}")

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
)

mlp = MLPClassifier(hidden_layer_sizes=(256, 128),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=42)

mlp.fit(X_train, y_train)

# Set a probability threshold for rejection
threshold = 0.6

# Predict probabilities on the test set
probs = mlp.predict_proba(X_test)
preds = mlp.predict(X_test)

# Apply threshold: if max(probability) < threshold, label as "Other" (-1)
final_preds = []
for prob, pred in zip(probs, preds):
    if np.max(prob) < threshold:
        final_preds.append(-1)
    else:
        final_preds.append(pred)
final_preds = np.array(final_preds)

# For evaluation, note that ground truth does not include "Other" (-1)
# So any sample flagged as -1 will count as a misclassification.
acc = accuracy_score(y_test, final_preds)
print("\nTest accuracy with rejection threshold:", acc)
print("\nClassification Report (classes: 0,1,2 and -1 for Other):")
print(classification_report(y_test, final_preds, labels=[0,1,2,-1]))