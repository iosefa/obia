import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
import laspy
from shapely.geometry import box
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


from sklearn.neural_network import MLPClassifier

INDIVIDUAL_CROWNS_DIR = "/Users/iosefa/trial_small/individual_crowns"
INDIVIDUAL_CROWNS_POINTS_DIR = "/Users/iosefa/trial_small/individual_crowns_points"
LABELS = "/Users/iosefa/trial_small/travis/labels.gpkg"

labels = gpd.read_file(LABELS)
labels = labels.to_crs(epsg=32605)

def code_to_class(code):
    if code == "METPOL":
        return 1
    elif code == "GREROB":
        return 2
    elif code == "ACAKOA":
        return 3
    elif code == "DIOSAN":
        return 4
    else:
        return 6

def min_max_scale_per_band(band):
    """
    Scale a single band (2D array) to [0..255] using min–max normalization.
    """
    min_val = band.min()
    max_val = band.max()
    if max_val == min_val:
        return np.zeros_like(band, dtype=np.float32)

    scaled = (band - min_val) / (max_val - min_val)
    scaled *= 255.0
    return scaled

class CrownDataset(Dataset):
    def __init__(self, records, img_transform=None, max_points=2048):
        self.records = records
        self.img_transform = img_transform
        self.max_points = max_points

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        tif_path, las_path, label = self.records[idx]

        with rasterio.open(tif_path) as src:
            img = src.read()

        if img.shape[0] == 9:
            img = img[[4, 2, 1], :, :]

        img = img.astype(np.float32)
        for b in range(img.shape[0]):
            img[b] = min_max_scale_per_band(img[b])
        img = np.moveaxis(img, 0, -1)   # shape: (H, W, C)
        img = img.astype(np.uint8)

        pil_img = Image.fromarray(img)
        if self.img_transform:
            pil_img = self.img_transform(pil_img)

        with laspy.open(las_path) as f:
            las = f.read()
            x = las.x
            y = las.y
            z = las.z
        points = np.vstack((x, y, z)).T  # shape: (N, 3)

        if points.shape[0] > self.max_points:
            choice = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[choice, :]

        points = torch.from_numpy(points).float()  # (N, 3)

        return {
            'image': pil_img,   # a Tensor if img_transform includes ToTensor
            'points': points,
            'label': label
        }

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            feat_size = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            feat_size = 2048
        else:
            raise ValueError("Unsupported model_name")
        self.model.fc = nn.Identity()
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.max(x, dim=1)[0]
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x

    @property
    def output_dim(self):
        return self._out_features

labels['class'] = labels['code'].apply(code_to_class)
labels = labels[["class", "geometry"]]

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

    candidate_points = labels[labels.geometry.within(tif_bbox)]
    if len(candidate_points) == 0:
        continue
    label = candidate_points["class"].mode()[0]
    training_records.append((tif_path, las_path, label))

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = CrownDataset(training_records, img_transform=img_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

image_feature_extractor = ResNetFeatureExtractor('resnet18', pretrained=True).to(device)
point_feature_extractor = PointNetFeatureExtractor(input_dim=3, hidden_dim=64).to(device)

image_feature_extractor.eval()
point_feature_extractor.eval()

X_features = []
y_labels = []

with torch.no_grad():
    for batch in dataloader:
        img = batch['image'].to(device)
        pts = batch['points'].to(device)
        label = batch['label'].item()

        img_feats = image_feature_extractor(img)
        pt_feats = point_feature_extractor(pts)

        combined_feat = torch.cat([img_feats, pt_feats], dim=1)

        combined_feat_np = combined_feat.cpu().numpy().squeeze(0)
        X_features.append(combined_feat_np)
        y_labels.append(label)

X_features = np.array(X_features)
y_labels = np.array(y_labels)

class_mapping = {1: 0, 2: 1, 4: 2, 6: 3}
y_mapped = np.array([class_mapping[l] for l in y_labels])

print("\nLabel distribution:")
for new_label in np.unique(y_mapped):
    print(f"New Class {new_label}: {np.sum(y_mapped == new_label)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_mapped, test_size=0.3, random_state=42, stratify=y_mapped
)

print(f"\nTrain samples: {len(y_train)}, Test samples: {len(y_test)}")

mlp = MLPClassifier(hidden_layer_sizes=(256, 128),
                    activation='relu',
                    solver='adam',
                    max_iter=10000,
                    random_state=42)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nTest accuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
