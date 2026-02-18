import os, glob, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import laspy
from shapely.geometry import box

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix

# ========= PATHS (EDIT THESE) =========
IMG_DIR = "/Users/iosefa/repos/sam/area51_ss1_crowns_img"
LAS_DIR = "/Users/iosefa/repos/sam/area51_ss1_crowns_las"

MODEL_PTH  = "/Users/iosefa/repos/sam/cnn_crowns_resnet18_mlp.pth"
MODEL_META = "/Users/iosefa/repos/sam/cnn_crowns_resnet18_mlp.meta.json"

# Optional: labeled validation GPKG with 0-indexed labels in 'feature_class'
# If you don't have this, set to None and the script will skip accuracy.
VALIDATION_GPKG = "/Users/iosefa/repos/sam/area51_labelled_validated_subset1.gpkg"  # or None

# Outputs
CROWNS_GPKG     = "/Users/iosefa/repos/sam/crowns.gpkg"
CROWNS_LAYER    = "crowns"
PRED_GPKG       = "/Users/iosefa/repos/sam/predicted_cnn.gpkg"
PRED_LAYER      = "predicted_cnn"

# ========= RUNTIME/DEVICE =========
def get_device_and_loader_opts():
    if torch.backends.mps.is_available():
        return torch.device("mps"), 0, False
    if torch.cuda.is_available():
        return torch.device("cuda"), 2, True
    return torch.device("cpu"), 0, False

# ========= MODEL (must mirror training) =========
class CombinedNet(nn.Module):
    def __init__(self, num_classes, freeze_resnet=False):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_resnet:
            for p in base.parameters():
                p.requires_grad = False
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.classifier = nn.Linear(512 + 64, num_classes)

    def forward(self, img, pts):
        x_img = self.backbone(img).view(img.size(0), -1)  # (B,512)
        h = F.relu(self.fc1(pts))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h, _ = torch.max(h, dim=1)                        # (B,256)
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))                           # (B,64)
        x = torch.cat([x_img, h], dim=1)                  # (B,576)
        return self.classifier(x)

# ========= IO / PREPROC =========
def read_image_as_3band_tensor(tif_path, band_sel_9, band_sel_8, img_size):
    with rasterio.open(tif_path) as src:
        img = src.read()  # (C,H,W)
    img = img.astype(np.float32)

    if img.shape[0] == 9 and band_sel_9:
        img = img[band_sel_9, :, :]
    elif img.shape[0] == 8 and band_sel_8:
        img = img[band_sel_8, :, :]
    elif img.shape[0] >= 3:
        img = img[:3, :, :]
    else:
        raise ValueError(f"Unexpected band count {img.shape[0]} in {tif_path}")

    img2 = []
    for c in range(img.shape[0]):
        band = img[c]
        finite = np.isfinite(band)
        if not np.any(finite):
            band = np.zeros_like(band, dtype=np.float32)
        else:
            lo, hi = np.percentile(band[finite], [2, 98])
            if hi <= lo:
                lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = 0.0, 1.0
            band = np.clip(band, lo, hi)
            band = (band - lo) / max(hi - lo, 1e-6)
        img2.append(band.astype(np.float32))
    img = np.stack(img2, axis=0)

    t = torch.from_numpy(img).unsqueeze(0)
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    t = (t - mean) / std
    return t.float()

def read_las_points(las_path):
    with laspy.open(las_path) as f:
        las = f.read()
        pts = np.vstack([las.x, las.y, las.z]).T
    return pts

def normalize_points_xy_center(points):
    xy = points[:, :2]
    z  = points[:, 2:3]
    xy_c = xy - xy.mean(axis=0, keepdims=True)
    std = xy_c.std(axis=0, keepdims=True) + 1e-6
    xy_n = xy_c / std
    z_n = (z - z.mean()) / (z.std() + 1e-6)
    return np.hstack([xy_n, z_n])

def pad_or_sample_points(points, n):
    n_pts = points.shape[0]
    if n_pts == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if n_pts > n:
        idx = np.random.choice(n_pts, n, replace=False)
        return points[idx, :].astype(np.float32)
    if n_pts < n:
        add = np.random.choice(n_pts, n - n_pts, replace=True)
        return np.vstack([points, points[add, :]]).astype(np.float32)
    return points.astype(np.float32)

class CrownDataset(Dataset):
    def __init__(self, records, meta):
        self.records = records
        self.meta = meta

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        tif_path, las_path = self.records[idx]
        img = read_image_as_3band_tensor(tif_path,
                                         self.meta["band_selection_9"],
                                         self.meta["band_selection_8"],
                                         self.meta["img_size"])
        pts = read_las_points(las_path)
        pts = normalize_points_xy_center(pts)
        pts = pad_or_sample_points(pts, self.meta["max_points"])
        pts = torch.from_numpy(pts)
        return img, pts, Path(tif_path).stem

# ========= BUILD RECORDS + GEOMS =========
def build_records_and_geoms(img_dir, las_dir):
    tif_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    las_files = sorted(glob.glob(os.path.join(las_dir, "*.las")) +
                       glob.glob(os.path.join(las_dir, "*.laz")))
    tif_dict = {Path(p).stem: p for p in tif_files}
    las_dict = {Path(p).stem: p for p in las_files}
    keys = sorted(set(tif_dict.keys()).intersection(las_dict.keys()))
    if not keys:
        raise RuntimeError("No matching *.tif with *.las(laz) basenames found.")

    recs, geoms, keys_ok, crs_out = [], [], [], None
    for k in keys:
        tif_path = tif_dict[k]
        las_path = las_dict[k]
        with rasterio.open(tif_path) as src:
            left, bottom, right, top = src.bounds
            crs_tif = src.crs
        if crs_tif is None:
            warnings.warn(f"{tif_path} has no CRS; skipping.")
            continue
        if crs_out is None:
            crs_out = crs_tif
        geom = box(left, bottom, right, top)
        recs.append((tif_path, las_path))
        geoms.append(geom)
        keys_ok.append(k)
    gdf = gpd.GeoDataFrame({"key": keys_ok}, geometry=geoms, crs=crs_out)
    return recs, gdf

# ========= MAIN =========
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    device, NUM_WORKERS, PIN_MEMORY = get_device_and_loader_opts()
    print(f"Device: {device} | workers={NUM_WORKERS} pin_memory={PIN_MEMORY}")

    # Load metadata + model
    with open(MODEL_META, "r") as f:
        meta = json.load(f)
    num_classes = int(meta["num_classes"])
    class_map = meta["class_map"]               # raw(0-index) -> contiguous id used by the model
    inv_class_map = {v: k for k, v in class_map.items()}  # model id -> raw(0-index) label

    model = CombinedNet(num_classes=num_classes, freeze_resnet=meta.get("freeze_resnet", False)).to(device)
    state = torch.load(MODEL_PTH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Build records + crowns geoms from the TIF footprints
    records, crowns_gdf = build_records_and_geoms(IMG_DIR, LAS_DIR)

    # Save crowns.gpkg
    if Path(CROWNS_GPKG).exists():
        try: os.remove(CROWNS_GPKG)
        except Exception: pass
    crowns_gdf.to_file(CROWNS_GPKG, layer=CROWNS_LAYER, driver="GPKG")
    print(f"Wrote crowns: {CROWNS_GPKG} (layer='{CROWNS_LAYER}') rows={len(crowns_gdf)}")

    # Dataset/DataLoader
    ds = CrownDataset(records, meta)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Inference
    keys, pred_ids, prob_rows = [], [], []
    with torch.no_grad():
        for img, pts, key in loader:
            img, pts = img.to(device), pts.to(device)
            logits = model(img, pts)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            pred_ids.extend(pred.tolist())
            prob_rows.extend(probs.tolist())
            keys.extend(list(key))

    # Build predictions GeoDataFrame
    pred_df = pd.DataFrame({"key": keys, "pred_id": pred_ids})
    for k in range(num_classes):
        pred_df[f"prob_{k}"] = np.array(prob_rows)[:, k]

    # Map model’s contiguous id back to raw (0-index) label for readability
    pred_df["predicted_class"] = pred_df["pred_id"].map(inv_class_map).astype(int)

    out_gdf = crowns_gdf.merge(pred_df, on="key", how="inner")

    # If validation labels are available, compute accuracy
    if VALIDATION_GPKG and Path(VALIDATION_GPKG).exists():
        labels_gdf = gpd.read_file(VALIDATION_GPKG)
        if labels_gdf.crs != out_gdf.crs:
            labels_gdf = labels_gdf.to_crs(out_gdf.crs)

        # spatial join (each crown box to intersecting labeled crown polygon)
        joined = gpd.sjoin(out_gdf, labels_gdf[["feature_class", "geometry"]], how="left", predicate="intersects")
        if "feature_class" in joined.columns:
            mask = joined["feature_class"].notna()
            y_true = joined.loc[mask, "feature_class"].astype(int).values
            y_pred = joined.loc[mask, "predicted_class"].astype(int).values
            print(f"\nValidation matches: {mask.sum()} of {len(joined)} crowns")
            if mask.sum() > 0:
                print("Classification Report (feature_class 0-indexed):")
                print(classification_report(y_true, y_pred, digits=4))
                print("Confusion Matrix:")
                print(confusion_matrix(y_true, y_pred))
            out_gdf = joined.drop(columns=["index_right"])
        else:
            print("⚠️  No 'feature_class' found in validation gpkg; skipping accuracy.")
    else:
        print("No validation GPKG provided or file missing; skipping accuracy.")

    # Write predictions GPKG (with predicted_class)
    if Path(PRED_GPKG).exists():
        try: os.remove(PRED_GPKG)
        except Exception: pass
    out_gdf.to_file(PRED_GPKG, layer=PRED_LAYER, driver="GPKG")
    print(f"\nWrote predictions: {PRED_GPKG} (layer='{PRED_LAYER}') rows={len(out_gdf)}")