# scratch_cnn_aoi_pruned.py
# Train a crowns classifier with classes 2 and 4 REMOVED (0-indexed raw labels).
# Includes: training-time norm stats + band order lock, optional focal loss,
# val confusion-matrix dump, and a DEFAULT AOI fine-tuning stage.
# + Best-checkpoint saving for pretrain and AOI.

import os, glob, warnings, random, json, collections
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box
from shapely.ops import transform as shp_transform
import pyproj

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models

# ---------- Optional EPT handler ----------
try:
    from pyforestscan.handlers import read_lidar  # used when LAS_DIR is None
except Exception:
    read_lidar = None

# ===============================
# CONFIG (pretrain)
# ===============================
TRAINING_GPKG = "/Users/iosefa/repos/sam/trainings_filtered.gpkg"
INDIVIDUAL_CROWNS_DIR = "/Users/iosefa/repos/sam/crowns_img"
INDIVIDUAL_CROWNS_POINTS_DIR = "/Users/iosefa/repos/sam/crowns_las"   # if None, uses EPT

# EPT fallback (used if INDIVIDUAL_CROWNS_POINTS_DIR is None or LAS for a key is missing)
EPT_JSON = "/Users/iosefa/Downloads/ept6635/ept.json"
EPT_SRS  = "EPSG:6635"   # CRS of the EPT dataset (must be correct)

MODEL_PTH  = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft1.pth"
MODEL_META = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft1.meta.json"

# IMPORTANT: classes (raw, 0-index) to DROP from training (your AOI never has these)
DROP_RAW_CLASSES = {2, 4}

MAX_POINTS = 2048           # only downsample if > MAX_POINTS
IMG_SIZE   = 320
BATCH_SIZE = 16
EPOCHS     = 30
PATIENCE   = 6              # early stop on val acc
WARMUP_FREEZE_EPOCHS = 2    # freeze backbone for first N epochs
LR_BACKBONE = 3e-5
LR_HEAD     = 8e-5
WD         = 1e-4
ADAMW_EPS  = 1e-8
RANDOM_SEED= 42
FREEZE_RESNET_INIT = True
VAL_SPLIT  = 0.1

# If your chips are 9- or 8-band stacks, pick 3 (WV3/WV2 examples below) — 0-based indices
BAND_SELECTION_9 = [4, 2, 1]   # WV3 NIR, Red, Green
BAND_SELECTION_8 = [6, 4, 2]   # WV2 NIR, Red, Green

# Image robustness
MIN_FINITE_PIX = 50

# Regularization
USE_MIXUP   = True
MIXUP_ALPHA = 0.3
LABEL_SMOOTH = 0.02

# Loss choice
USE_FOCAL = True   # toggle: True -> FocalLoss, False -> LabelSmoothingCE
FOCAL_GAMMA = 1.5

# ===============================
# CONFIG (AOI fine-tune — DEFAULT: enabled)
# ===============================
AOI_GPKG               = "/Users/iosefa/repos/sam/overlap_labels.gpkg"  # same labels you use for priors/eval
AOI_LABEL_COLUMN       = "class"   # 1-indexed raw labels in your GPKG
AOI_VAL_SPLIT          = 0.0       # fine-tune is usually small; do full-train by default
AOI_EPOCHS             = 10
AOI_PATIENCE           = 3
AOI_BATCH_SIZE         = 12
# smaller AOI LR (helps convergence without overfitting)
AOI_LR_BACKBONE        = 5e-6
AOI_LR_HEAD            = 1.5e-5
AOI_WD                 = 5e-5
AOI_USE_MIXUP          = False     # keep it off for fine-tune
AOI_LABEL_SMOOTH       = 0.01
AOI_USE_FOCAL          = True
AOI_FOCAL_GAMMA        = 1.25
AOI_ENABLE             = True      # default ON (will silently skip if GPKG missing)
AOI_MIN_SAMPLES        = 20        # require at least this many samples post-prune to run

# ===============================
# Reproducibility
# ===============================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def dev():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

# ===============================
# READ & MATCH TIF/LAS + LABELS (PRETRAIN)
# ===============================
gdf_labels = gpd.read_file(TRAINING_GPKG)
# convert to raw 0-index labels in your file
gdf_labels["feature_class"] = pd.to_numeric(gdf_labels["class"], errors="coerce").astype("Int64") - 1
gdf_labels = gdf_labels.dropna(subset=["feature_class"])

tif_files = sorted(glob.glob(os.path.join(INDIVIDUAL_CROWNS_DIR, "*.tif")))
las_files = sorted(glob.glob(os.path.join(INDIVIDUAL_CROWNS_POINTS_DIR, "*.las"))) if INDIVIDUAL_CROWNS_POINTS_DIR else []
las_all   = las_files

tif_dict = {Path(p).stem: p for p in tif_files}
las_dict = {Path(p).stem: p for p in las_all}

keys_all = sorted(tif_dict.keys())

records_all, geoms_all, labels_raw_all, keys_ok_all = [], [], [], []

if gdf_labels.crs is None or gdf_labels.crs.is_geographic:
    gdf_labels = gdf_labels.to_crs(6635)

for key in keys_all:
    tif_path = tif_dict[key]
    las_path = las_dict.get(key, None)

    with rasterio.open(tif_path) as src:
        left, bottom, right, top = src.bounds
        crs_tif = src.crs
        if crs_tif is None:
            warnings.warn(f"{tif_path} has no CRS; skipping.")
            continue

    chip_geom = box(left, bottom, right, top)
    chip_gdf  = gpd.GeoDataFrame({"key":[key]}, geometry=[chip_geom], crs=crs_tif)
    if chip_gdf.crs != gdf_labels.crs:
        chip_gdf = chip_gdf.to_crs(gdf_labels.crs)
        chip_geom = chip_gdf.geometry.iloc[0]

    hits = gdf_labels[gdf_labels.geometry.intersects(chip_geom)]
    if hits.empty:
        continue

    inter_areas = hits.geometry.intersection(chip_geom).area
    idx = inter_areas.idxmax()
    raw_label = hits.loc[idx, "feature_class"]
    if pd.isna(raw_label):
        continue

    raw_label = int(raw_label)
    # collect all, we will filter after
    records_all.append((tif_path, las_path, raw_label, key, hits.loc[idx, "geometry"]))
    geoms_all.append(hits.loc[idx, "geometry"])
    labels_raw_all.append(raw_label)
    keys_ok_all.append(key)

print(f"Matched crowns with label & geometry: {len(records_all)} (before pruning)")

# ---- PRUNE classes you don't want in AOI ----
mask_keep = [y not in DROP_RAW_CLASSES for y in labels_raw_all]
records = [rec for rec, keep in zip(records_all, mask_keep) if keep]
geoms   = [g   for g,   keep in zip(geoms_all,   mask_keep) if keep]
labels_raw = [y for y,   keep in zip(labels_raw_all, mask_keep) if keep]
keys_ok = [k  for k,   keep in zip(keys_ok_all,  mask_keep) if keep]

print(f"Kept after pruning classes {sorted(DROP_RAW_CLASSES)}: {len(records)}")

gdf_records = gpd.GeoDataFrame({"key": keys_ok, "raw_label": labels_raw},
                               geometry=geoms, crs=gdf_labels.crs)

# Build contiguous class map for *remaining* raw labels
RAW_LABELS_KEPT = sorted(set(labels_raw))
CLASS_MAP = {lbl: i for i, lbl in enumerate(RAW_LABELS_KEPT)}  # raw->contiguous
INV_CLASS_MAP = {i: lbl for lbl, i in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)

print(f"Raw labels kept: {RAW_LABELS_KEPT}  -> contiguous ids 0..{NUM_CLASSES-1}")
print("Class histogram (raw kept):", dict(collections.Counter(labels_raw)))
print("Class histogram (contiguous ids):", dict(collections.Counter([CLASS_MAP[y] for y in labels_raw])))

# ===============================
# TRAINING-TIME NORM STATS & BAND LOCK
# ===============================
RUN_STATS = {
    "sum":   np.zeros(3, dtype=np.float64),
    "sum2":  np.zeros(3, dtype=np.float64),
    "count": np.zeros(3, dtype=np.int64),
}
EXPECTED_BAND_IDX = None
if BAND_SELECTION_9 is not None:
    EXPECTED_BAND_IDX = BAND_SELECTION_9
elif BAND_SELECTION_8 is not None:
    EXPECTED_BAND_IDX = BAND_SELECTION_8

# ===============================
# IO / PREPROC
# ===============================
def _apply_nodata_to_nan(arr, src):
    arr = arr.astype(np.float32, copy=False)
    if src.nodata is not None and np.isfinite(src.nodata):
        arr[arr == src.nodata] = np.nan
    return arr

def _augment_image_cheap(img):
    # img: (C,H,W) in [0,1] roughly after percentile norm
    if random.random() < 0.5:
        img = img[:, :, ::-1].copy()
    if random.random() < 0.5:
        img = img[:, ::-1, :].copy()
    C = img.shape[0]
    for c in range(C):
        g = 1.0 + np.random.uniform(-0.15, 0.15)
        b = np.random.uniform(-0.07, 0.07)
        img[c] = np.clip(img[c]*g + b, 0.0, 1.0)
    return img

def read_image_as_3band_tensor(tif_path, training=False):
    with rasterio.open(tif_path) as src:
        img = src.read()                            # (C,H,W)
        img = _apply_nodata_to_nan(img, src)

    C = img.shape[0]
    if C == 9 and BAND_SELECTION_9:
        img = img[BAND_SELECTION_9, :, :]
    elif C == 8 and BAND_SELECTION_8:
        img = img[BAND_SELECTION_8, :, :]
    elif C >= 3:
        img = img[:3, :, :]
    else:
        raise ValueError(f"Unexpected band count {C} in {tif_path}")

    img2 = []
    for c in range(img.shape[0]):
        band = img[c]
        finite = np.isfinite(band)
        if not np.any(finite):
            img2.append(np.zeros_like(band, dtype=np.float32)); continue
        lo, hi = np.nanpercentile(band[finite], [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
        band = np.clip(band, lo, hi)
        band = (band - lo) / max(hi - lo, 1e-6)
        img2.append(band.astype(np.float32))
    img = np.stack(img2, axis=0)

    if training:
        img = _augment_image_cheap(img)

    t = torch.from_numpy(img).unsqueeze(0)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)

    t[~torch.isfinite(t)] = 0.0

    # -------- training-time running stats (post percentile norm) --------
    if training:
        with torch.no_grad():
            for c in range(min(3, t.shape[0])):
                band_np = t[c].cpu().numpy()
                m = float(np.nanmean(band_np))
                v = float(np.nanvar(band_np))
                n = band_np.size
                RUN_STATS["sum"][c]  += m * n
                RUN_STATS["sum2"][c] += (v + m*m) * n   # E[x^2] * n
                RUN_STATS["count"][c]+= n

    # -------- use persisted normalization if available (else ImageNet fallback) --------
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]
    try:
        if os.path.exists(MODEL_META):
            with open(MODEL_META, "r") as _f:
                _meta_prev = json.load(_f)
                if "norm_mean" in _meta_prev and "norm_std" in _meta_prev:
                    norm_mean = _meta_prev["norm_mean"]
                    norm_std  = _meta_prev["norm_std"]
    except Exception:
        pass

    mean = torch.tensor(norm_mean, dtype=torch.float32).view(-1,1,1)
    std  = torch.tensor(norm_std,  dtype=torch.float32).view(-1,1,1)
    if mean.shape[0] != t.shape[0]:
        mean = mean[:t.shape[0]]
        std  = std[:t.shape[0]]

    t = (t - mean) / (std.clamp_min(1e-6))
    return t.float()

def _pdal_struct_to_xyz(pc_obj):
    arr = pc_obj[0] if isinstance(pc_obj, (list, tuple)) else pc_obj
    if arr is None:
        return np.zeros((0,3), dtype=np.float32)
    if getattr(arr, "dtype", None) is None or arr.dtype.names is None:
        a = np.asarray(arr)
        return a[:, :3].astype(np.float32) if a.ndim == 2 and a.shape[1] >= 3 else np.zeros((0,3), dtype=np.float32)
    names = [n.lower() for n in arr.dtype.names]
    if all(n in names for n in ("x","y","z")):
        X = arr[arr.dtype.names[names.index("x")]]
        Y = arr[arr.dtype.names[names.index("y")]]
        Z = arr[arr.dtype.names[names.index("z")]]
    else:
        cols = arr.dtype.names[:3]
        X = arr[cols[0]]; Y = arr[cols[1]]; Z = arr[cols[2]]
    return np.stack([np.asarray(X), np.asarray(Y), np.asarray(Z)], axis=1).astype(np.float32)

def _reproject_points(xyz, src_crs, dst_crs):
    if xyz.shape[0] == 0 or src_crs == dst_crs:
        return xyz
    fwd = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    xs, ys = fwd(xyz[:,0], xyz[:,1])
    out = xyz.copy()
    out[:,0] = xs
    out[:,1] = ys
    return out

def read_points_for_training(las_path, crown_poly_in_label_crs, label_crs):
    # A) LAS
    if las_path is not None and Path(las_path).exists():
        import laspy
        with laspy.open(las_path) as f:
            las = f.read()
            pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        return pts
    # B) EPT fallback
    if read_lidar is None or not Path(EPT_JSON).exists():
        return np.zeros((0,3), dtype=np.float32)
    try:
        label_crs = pyproj.CRS.from_user_input(label_crs)
        ept_crs   = pyproj.CRS.from_user_input(EPT_SRS)
    except Exception:
        return np.zeros((0,3), dtype=np.float32)
    if label_crs != ept_crs:
        to_ept = pyproj.Transformer.from_crs(label_crs, ept_crs, always_xy=True).transform
        poly_q = shp_transform(to_ept, crown_poly_in_label_crs)
    else:
        poly_q = crown_poly_in_label_crs
    minx, miny, maxx, maxy = poly_q.bounds
    bounds = (minx, miny, maxx, maxy)
    try:
        pc = read_lidar(EPT_JSON, EPT_SRS, bounds, crop_poly=True, poly=poly_q.wkt)
        xyz = _pdal_struct_to_xyz(pc)
        if xyz.shape[0] > 0 and (label_crs != ept_crs):
            xyz = _reproject_points(xyz, ept_crs, label_crs)
        return xyz.astype(np.float32)
    except Exception:
        return np.zeros((0,3), dtype=np.float32)

def normalize_points_xy_center(points):
    if points.size == 0:
        return points
    xy = points[:, :2]
    z  = points[:, 2:3]
    xy_c = xy - xy.mean(axis=0, keepdims=True)
    std = xy_c.std(axis=0, keepdims=True)
    std[std == 0] = 1e-6
    xy_n = xy_c / std
    z_std = z.std()
    z_std = z_std if np.isfinite(z_std) and z_std > 0 else 1e-6
    z_n = (z - z.mean()) / z_std
    out = np.hstack([xy_n, z_n]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out

def downsample_if_needed(points, n=MAX_POINTS):
    n_pts = points.shape[0]
    if n_pts == 0:
        return points
    if n_pts > n:
        idx = np.random.choice(n_pts, n, replace=False)
        out = points[idx, :]
    else:
        out = points
    out = out.astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out

# ===============================
# DATASET
# ===============================
class CrownDataset(Dataset):
    def __init__(self, recs, class_map, label_crs, training=False):
        self.recs = recs
        self.class_map = class_map
        self.label_crs = label_crs
        self.training = training

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        tif_path, las_path, y_raw, key, crown_geom = self.recs[idx]
        y = self.class_map[y_raw]

        # image
        img = read_image_as_3band_tensor(tif_path, training=self.training)
        img[~torch.isfinite(img)] = 0.0

        # points
        pts = read_points_for_training(las_path, crown_geom, self.label_crs)
        pts = normalize_points_xy_center(pts)
        pts = downsample_if_needed(pts, MAX_POINTS)  # keep variable length

        # augs (training only)
        if self.training and pts.shape[0] > 0:
            # random small rotation of points
            theta = np.random.uniform(0.0, 2.0*np.pi)
            c, s = np.cos(theta), np.sin(theta)
            xy = pts[:, :2].copy()
            pts[:, 0] = c * xy[:, 0] - s * xy[:, 1]
            pts[:, 1] = s * xy[:, 0] + c * xy[:, 1]

            # random dropout & jitter
            keep_ratio = np.random.uniform(0.7, 1.0)
            n = pts.shape[0]
            k = max(1, int(n * keep_ratio))
            idx_keep = np.random.choice(n, k, replace=False)
            pts = pts[idx_keep, :]
            pts[:, 0:2] += np.random.normal(0.0, 0.02, size=(pts.shape[0], 2)).astype(np.float32)
            pts[:, 2:3] += np.random.normal(0.0, 0.03, size=(pts.shape[0], 1)).astype(np.float32)

            # flips aligned to image
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])  # horizontal
                pts[:, 0] = -pts[:, 0]
            if random.random() < 0.5:
                img = torch.flip(img, dims=[1])  # vertical
                pts[:, 1] = -pts[:, 1]

        pts_t = torch.from_numpy(pts.astype(np.float32))
        length = torch.tensor(pts_t.shape[0], dtype=torch.long)
        return img, pts_t, length, y, key

# ===============================
# MODEL (keep head shapes compatible with classifier script)
# ===============================
class CombinedNet(nn.Module):
    def __init__(self, num_classes, freeze_resnet=True):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        if freeze_resnet:
            for p in base.parameters():
                p.requires_grad = False
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1) for RN34

        self.fc1 = nn.Linear(3, 64, bias=True)
        self.fc2 = nn.Linear(64, 128, bias=True)
        self.fc3 = nn.Linear(128, 256, bias=True)  # time-dim features (T,256)

        # Max pool -> 256-d vector; keep fc4 input=256 to match your classifier script
        self.fc4 = nn.Linear(256, 128, bias=True)
        self.fc5 = nn.Linear(128, 64, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(512 + 64, num_classes)  # 512 (img RN34) + 64 (pts)

    def forward(self, img, pts, lengths=None):
        x_img = self.backbone(img).view(img.size(0), -1)     # (B,512)

        h = F.relu(self.fc1(pts))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))                               # (B,T,256)

        if lengths is not None:
            B, T, C = h.shape
            ar = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
            mask = ar >= lengths.unsqueeze(1)
            h = h.masked_fill(mask.unsqueeze(-1), float("-inf"))

        h, _ = torch.max(h, dim=1)                            # (B,256)
        h[~torch.isfinite(h)] = 0.0
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))                               # (B,64)

        x = torch.cat([x_img, h], dim=1)                      # (B,576)
        x = self.dropout(x)
        return self.classifier(x)

# ===============================
# Collate (variable-length padding per batch)
# ===============================
def collate_variable(batch):
    imgs, pts_list, lengths_list, ys, keys = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    lengths = torch.stack(lengths_list, dim=0)
    Tmax = int(lengths.max().item()) if len(lengths_list) else 0
    if Tmax == 0:
        pts_padded = torch.zeros((len(batch), 1, 3), dtype=torch.float32)
        lengths = torch.zeros((len(batch),), dtype=torch.long)
    else:
        pts_padded = torch.zeros((len(batch), Tmax, 3), dtype=torch.float32)
        for i, pts in enumerate(pts_list):
            n = pts.shape[0]
            if n > 0:
                pts_padded[i, :n, :] = pts
    ys = torch.tensor(ys, dtype=torch.long)
    return imgs, pts_padded, lengths, ys, list(keys)

# ===============================
# Helpers
# ===============================
def dev_loader_opts():
    if torch.backends.mps.is_available():
        return torch.device("mps"), 0, False
    if torch.cuda.is_available():
        return torch.device("cuda"), 2, True
    return torch.device("cpu"), 0, False

class LabelSmoothingCE(nn.Module):
    def __init__(self, weight=None, smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.smoothing = float(smoothing)
    def forward(self, logits, target):
        if self.smoothing <= 0.0:
            return F.cross_entropy(logits, target, weight=self.weight)
        num_classes = logits.size(-1)
        logprobs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = - (true_dist * logprobs)
        if self.weight is not None:
            loss = loss * self.weight.view(1, -1)
        return loss.sum(dim=1).mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.smoothing = smoothing
    def forward(self, logits, target):
        n_classes = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        p = torch.exp(logp)
        focal = (1 - p) ** self.gamma
        loss = - focal * true_dist * logp
        if self.weight is not None:
            loss = loss * self.weight.view(1, -1)
        return loss.sum(dim=1).mean()

def maybe_mixup_logits(logits, y, alpha):
    if alpha <= 0:
        return logits, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B = logits.size(0)
    idx = torch.randperm(B, device=logits.device)
    logits_mix = lam * logits + (1 - lam) * logits[idx]
    return logits_mix, (y, y[idx]), lam

def mixup_loss(logits_mix, y_pair, lam, criterion):
    y1, y2 = y_pair
    return lam * criterion(logits_mix, y1) + (1 - lam) * criterion(logits_mix, y2)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for img, pts, lengths, y, _ in loader:
        img, pts, lengths = img.to(device), pts.to(device), lengths.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        img[~torch.isfinite(img)] = 0.0
        pts[~torch.isfinite(pts)] = 0.0
        logits = model(img, pts, lengths)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
    return loss_sum / max(total,1), correct / max(total,1), cm

def run_epoch(model, loader, criterion, optimizer=None, device=torch.device("cpu"), use_mixup=True):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for img, pts, lengths, y, keys in loader:
        img, pts, lengths = img.to(device), pts.to(device), lengths.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        img[~torch.isfinite(img)] = 0.0
        pts[~torch.isfinite(pts)] = 0.0

        logits = model(img, pts, lengths)

        if use_mixup and train:
            logits_mix, y_pair, lam = maybe_mixup_logits(logits, y, MIXUP_ALPHA)
            loss = mixup_loss(logits_mix, y_pair, lam, criterion)
            pred = logits_mix.argmax(dim=1)
        else:
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total,1), correct / max(total,1)

def pretty_cm(cm):
    with np.printoptions(linewidth=120):
        return "\n" + "\n".join([" ".join([f"{v:5d}" for v in row]) for row in cm])

def save_state_dict(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))

# ===============================
# AOI fine-tune helpers
# ===============================
def build_aoi_records(aoi_gpkg, label_col, dropped_raw_classes, tif_dict, las_dict, target_crs):
    if not Path(aoi_gpkg).exists():
        return []
    g = gpd.read_file(aoi_gpkg)
    if label_col not in g.columns:
        return []
    # to raw-0-index
    g["feature_class"] = pd.to_numeric(g[label_col], errors="coerce").astype("Int64") - 1
    g = g.dropna(subset=["feature_class"])
    # align CRS
    if g.crs is None:
        try:
            g = g.set_crs(target_crs)
        except Exception:
            pass
    if g.crs != target_crs:
        g = g.to_crs(target_crs)

    # spatially match to chips by intersection and take max overlap (like pretrain)
    recs = []
    for key, tif_path in tif_dict.items():
        with rasterio.open(tif_path) as src:
            chip_geom = box(*src.bounds)
            chip_gdf  = gpd.GeoDataFrame({"key":[key]}, geometry=[chip_geom], crs=src.crs)
            if chip_gdf.crs != g.crs:
                chip_gdf = chip_gdf.to_crs(g.crs)
                chip_geom = chip_gdf.geometry.iloc[0]
        hits = g[g.geometry.intersects(chip_geom)]
        if hits.empty:
            continue
        inter_areas = hits.geometry.intersection(chip_geom).area
        idx = inter_areas.idxmax()
        raw_label = int(hits.loc[idx, "feature_class"])
        if raw_label in dropped_raw_classes:
            continue
        las_path = las_dict.get(key, None)
        recs.append((tif_dict[key], las_path, raw_label, key, hits.loc[idx, "geometry"]))
    return recs

def make_loader(recs, class_map, crs_str, batch_size, shuffle, sampler=None, workers=0, pin=False):
    ds = CrownDataset(recs, class_map, crs_str, training=shuffle)
    if sampler is not None:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler,
                            num_workers=workers, pin_memory=pin, collate_fn=collate_variable)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=workers, pin_memory=pin, collate_fn=collate_variable)
    return ds, loader

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    device, NUM_WORKERS, PIN_MEMORY = dev_loader_opts()
    print(f"Using device: {device}. workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}")
    if INDIVIDUAL_CROWNS_POINTS_DIR is None:
        print(f"EPT fallback mode: reader={'ok' if read_lidar is not None else 'MISSING'}, "
              f"ept_json_exists={Path(EPT_JSON).exists()}, ept_srs='{EPT_SRS}'")

    # ----- stratified split after pruning (pretrain) -----
    rng = np.random.RandomState(RANDOM_SEED)
    if VAL_SPLIT > 0.0 and len(records) >= 10:
        y_all = np.array([CLASS_MAP[y] for _,_,y,_,_ in records], dtype=np.int64)
        val_idx, train_idx = [], []
        for c in range(NUM_CLASSES):
            cls_idx = np.where(y_all == c)[0]
            rng.shuffle(cls_idx)
            n_val_c = int(np.round(len(cls_idx) * VAL_SPLIT))
            val_idx.extend(cls_idx[:n_val_c].tolist())
            train_idx.extend(cls_idx[n_val_c:].tolist())
        rng.shuffle(val_idx); rng.shuffle(train_idx)
        train_recs = [records[i] for i in train_idx]
        val_recs   = [records[i] for i in val_idx]
    else:
        train_recs = records
        val_recs   = []

    # ----- class weights & sampler on train only -----
    y_train_ids = np.array([CLASS_MAP[y] for _,_,y,_,_ in train_recs], dtype=np.int64)
    counts = np.bincount(y_train_ids, minlength=NUM_CLASSES).astype(np.float64)
    inv_freq = 1.0 / np.clip(counts, 1, None)
    ce_weights_np = inv_freq / (inv_freq.mean() + 1e-12)
    ce_weight = torch.tensor(ce_weights_np, dtype=torch.float32, device=device)

    # priors (for info only)
    priors = (counts + 1.0) / (counts.sum() + NUM_CLASSES)  # Laplace
    log_priors = np.log(priors + 1e-12)

    print("Train class counts (pruned):", counts.tolist())
    print("CE class weights:", ce_weights_np.tolist())
    print("Log priors (pruned):", log_priors.tolist())

    sample_weights = ce_weights_np[y_train_ids]
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_ds = CrownDataset(train_recs, CLASS_MAP, str(gdf_labels.crs), training=True)
    val_ds   = CrownDataset(val_recs, CLASS_MAP, str(gdf_labels.crs), training=False) if val_recs else None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_variable)
    val_loader = (DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             collate_fn=collate_variable)
                  if val_ds else None)

    print(f"Train size (pruned): {len(train_ds)} | Val size: {len(val_ds) if val_ds else 0}")

    # Quick sanity: zero-point examples
    zero_pts = 0
    for _, (tif_path, las_path, y_raw, key, poly) in enumerate(train_recs[:100]):
        pts = read_points_for_training(las_path, poly, str(gdf_labels.crs))
        if pts.shape[0] == 0:
            zero_pts += 1
    print(f"Sampled train diagnostics: ~{zero_pts} / 100 examples had 0 LiDAR points.")

    # ----- Model / Loss / Optim (pretrain) -----
    model = CombinedNet(num_classes=NUM_CLASSES, freeze_resnet=FREEZE_RESNET_INIT).to(device)

    # Initialize classifier bias with pruned log-priors (stabilizes early training)
    with torch.no_grad():
        model.classifier.bias.copy_(torch.tensor(log_priors, dtype=torch.float32, device=device))

    if USE_FOCAL:
        criterion = FocalLoss(weight=ce_weight, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTH)
    else:
        criterion = LabelSmoothingCE(weight=ce_weight, smoothing=LABEL_SMOOTH)

    # Two LR groups: backbone (after unfreeze) and heads
    head_params = list(model.fc1.parameters()) + list(model.fc2.parameters()) + \
                  list(model.fc3.parameters()) + list(model.fc4.parameters()) + \
                  list(model.fc5.parameters()) + list(model.classifier.parameters())
    backbone_params = list(model.backbone.parameters())

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WD},
    ], eps=ADAMW_EPS)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))

    best_val = -1.0
    best_state = None
    no_improve = 0
    best_pretrain_path = Path(MODEL_PTH).with_suffix(".best_pretrain.pth")
    best_pretrain_cm   = Path(MODEL_PTH).with_suffix(".val_cm.csv")

    for epoch in range(1, EPOCHS+1):
        # Unfreeze backbone after warmup
        if epoch == WARMUP_FREEZE_EPOCHS + 1 and FREEZE_RESNET_INIT:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WD})
            print(f"Epoch {epoch}: Unfroze backbone with LR={LR_BACKBONE:g}")

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, use_mixup=USE_MIXUP)

        if val_loader is not None and len(val_ds) > 0:
            val_loss, val_acc, cm = eval_epoch(model, val_loader, criterion, device, NUM_CLASSES)
            print(f"Epoch [{epoch:02d}/{EPOCHS}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\nConfusion matrix:\n{pretty_cm(cm)}")
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                no_improve = 0
                try:
                    np.savetxt(best_pretrain_cm, cm, fmt="%d", delimiter=",")
                except Exception:
                    pass
                # save best pretrain immediately
                save_state_dict(best_state, best_pretrain_path)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} (best Acc={best_val:.4f}).")
                    break
        else:
            print(f"Epoch [{epoch:02d}/{EPOCHS}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f}")

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best val checkpoint (Acc={best_val:.4f}).")

    # ===============================
    # AOI FINE-TUNE (DEFAULT ON, skips gracefully if not available)
    # ===============================
    ran_aoi = False
    best_aoi = -1.0
    best_aoi_state = None
    best_aoi_path = Path(MODEL_PTH).with_suffix(".best_aoi.pth")
    best_aoi_cm   = Path(MODEL_PTH).with_suffix(".aoi_val_cm.csv")

    if AOI_ENABLE and Path(AOI_GPKG).exists():
        print("\n=== AOI fine-tune stage ===")
        # Build AOI records by spatial match against same chip sources
        aoi_recs = build_aoi_records(AOI_GPKG, AOI_LABEL_COLUMN, DROP_RAW_CLASSES,
                                     tif_dict, las_dict, gdf_labels.crs)
        if len(aoi_recs) >= AOI_MIN_SAMPLES:
            # Check labels compatible with pretrain map
            aoi_raw = [y for _,_,y,_,_ in aoi_recs]
            aoi_kept = sorted(set(aoi_raw))
            if any(lbl not in CLASS_MAP for lbl in aoi_kept):
                raise RuntimeError(f"AOI labels {aoi_kept} include classes not present in pretrain map {sorted(CLASS_MAP.keys())}")
            # Use same CLASS_MAP so checkpoint stays compatible
            y_ids = np.array([CLASS_MAP[y] for _,_,y,_,_ in aoi_recs], dtype=np.int64)
            counts_aoi = np.bincount(y_ids, minlength=NUM_CLASSES).astype(np.float64)
            inv_freq_aoi = 1.0 / np.clip(counts_aoi, 1, None)
            ce_weights_aoi = inv_freq_aoi / (inv_freq_aoi.mean() + 1e-12)
            ce_weight_aoi = torch.tensor(ce_weights_aoi, dtype=torch.float32, device=device)

            print("AOI class counts (contiguous):", counts_aoi.tolist())
            print("AOI CE weights:", ce_weights_aoi.tolist())

            # Sampler (optional); for small sets, simple shuffle also works. We'll keep shuffle.
            aoi_ds, aoi_loader = make_loader(aoi_recs, CLASS_MAP, str(gdf_labels.crs),
                                             AOI_BATCH_SIZE, shuffle=True,
                                             sampler=None, workers=NUM_WORKERS, pin=PIN_MEMORY)

            # Optim/loss (low LR, all params trainable)
            for p in model.backbone.parameters():
                p.requires_grad = True
            params_all = [
                {"params": list(model.backbone.parameters()), "lr": AOI_LR_BACKBONE, "weight_decay": AOI_WD},
                {"params": list(model.fc1.parameters()) + list(model.fc2.parameters()) +
                           list(model.fc3.parameters()) + list(model.fc4.parameters()) +
                           list(model.fc5.parameters()) + list(model.classifier.parameters()),
                 "lr": AOI_LR_HEAD, "weight_decay": AOI_WD},
            ]
            optimizer_aoi = torch.optim.AdamW(params_all, eps=ADAMW_EPS)

            if AOI_USE_FOCAL:
                criterion_aoi = FocalLoss(weight=ce_weight_aoi, gamma=AOI_FOCAL_GAMMA, smoothing=AOI_LABEL_SMOOTH)
            else:
                criterion_aoi = LabelSmoothingCE(weight=ce_weight_aoi, smoothing=AOI_LABEL_SMOOTH)

            scheduler_aoi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_aoi, T_max=max(AOI_EPOCHS, 1))

            no_imp_aoi = 0
            # If AOI_VAL_SPLIT>0, we’ll build loaders once outside the loop
            if AOI_VAL_SPLIT > 0.0 and len(aoi_ds) >= 10:
                n = len(aoi_recs)
                idx = np.arange(n)
                np.random.shuffle(idx)
                n_val = int(round(n * AOI_VAL_SPLIT))
                val_idx = idx[:n_val].tolist()
                tr_idx  = idx[n_val:].tolist()
                aoi_tr = [aoi_recs[i] for i in tr_idx]
                aoi_va = [aoi_recs[i] for i in val_idx]
                _, aoi_tr_loader = make_loader(aoi_tr, CLASS_MAP, str(gdf_labels.crs),
                                               AOI_BATCH_SIZE, shuffle=True, workers=NUM_WORKERS, pin=PIN_MEMORY)
                _, aoi_va_loader = make_loader(aoi_va, CLASS_MAP, str(gdf_labels.crs),
                                               AOI_BATCH_SIZE, shuffle=False, workers=NUM_WORKERS, pin=PIN_MEMORY)
            else:
                aoi_tr_loader = aoi_loader
                aoi_va_loader = None

            for epoch in range(1, AOI_EPOCHS+1):
                tr_loss, tr_acc = run_epoch(model, aoi_tr_loader, criterion_aoi, optimizer_aoi,
                                            device, use_mixup=AOI_USE_MIXUP)

                if aoi_va_loader is not None:
                    val_loss, val_acc, cm_aoi = eval_epoch(model, aoi_va_loader, criterion_aoi, device, NUM_CLASSES)
                    print(f"[AOI {epoch:02d}/{AOI_EPOCHS}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
                    monitor = val_acc
                else:
                    print(f"[AOI {epoch:02d}/{AOI_EPOCHS}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f}")
                    cm_aoi = None
                    monitor = tr_acc

                if monitor > best_aoi:
                    best_aoi = monitor
                    best_aoi_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    no_imp_aoi = 0
                    # Save best AOI immediately
                    save_state_dict(best_aoi_state, best_aoi_path)
                    if cm_aoi is not None:
                        try:
                            np.savetxt(best_aoi_cm, cm_aoi, fmt="%d", delimiter=",")
                        except Exception:
                            pass
                else:
                    no_imp_aoi += 1
                    if no_imp_aoi >= AOI_PATIENCE:
                        print(f"AOI early stopping at epoch {epoch} (best Acc={best_aoi:.4f}).")
                        break

                scheduler_aoi.step()

            if best_aoi_state is not None:
                model.load_state_dict(best_aoi_state)
                print(f"Loaded best AOI checkpoint (Acc={best_aoi:.4f}).")
            ran_aoi = True
        else:
            print(f"AOI fine-tune skipped: found {len(aoi_recs)} samples (< {AOI_MIN_SAMPLES})")
    else:
        print("AOI fine-tune disabled or GPKG missing; skipping.")

    # ----- Compute and persist final per-band norm stats (from RUN_STATS) -----
    if RUN_STATS["count"].sum() > 0:
        total = RUN_STATS["count"].astype(np.float64)
        ex1   = RUN_STATS["sum"] / np.clip(total, 1, None)
        ex2   = RUN_STATS["sum2"] / np.clip(total, 1, None)
        var   = np.maximum(0.0, ex2 - ex1**2)
        norm_mean = ex1.tolist()
        norm_std  = np.sqrt(var).tolist()
    else:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std  = [0.229, 0.224, 0.225]

    # ----- Save -----
    # Final (post-AOI if ran, otherwise best pretrain already loaded)
    torch.save(model.state_dict(), MODEL_PTH)

    meta = {
        "class_map": CLASS_MAP,                   # raw->contiguous (post-prune)
        "inverse_class_map": INV_CLASS_MAP,       # contiguous->raw
        "raw_labels_kept": RAW_LABELS_KEPT,       # e.g., [0,1,3,5]
        "dropped_raw_labels": sorted(DROP_RAW_CLASSES),  # [2,4]
        "num_classes": NUM_CLASSES,               # 4
        "img_size": IMG_SIZE,
        "max_points": MAX_POINTS,
        "band_selection_9": BAND_SELECTION_9,
        "band_selection_8": BAND_SELECTION_8,
        # new fields consumed by the inference script:
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "expected_band_idx": (BAND_SELECTION_9 if BAND_SELECTION_9 is not None else BAND_SELECTION_8),
        "freeze_resnet": False,
        "architecture": "resnet34+pointMLP(maxpool)",  # fc4 in_features=256 (non-dual-pool)
        "weights": "IMAGENET1K_V1",
        "class_weights": ce_weights_np.tolist(),
        "val_split": VAL_SPLIT,
        "random_seed": RANDOM_SEED,
        "points_variable_length": True,
        "log_priors": log_priors.tolist(),
        "label_smoothing": LABEL_SMOOTH,
        "use_focal": USE_FOCAL,
        "focal_gamma": FOCAL_GAMMA,
        "aoi_finetuned": ran_aoi,
        "aoi_cfg": {
            "aoi_gpkg": AOI_GPKG,
            "epochs": AOI_EPOCHS,
            "lr_backbone": AOI_LR_BACKBONE,
            "lr_head": AOI_LR_HEAD,
            "use_mixup": AOI_USE_MIXUP,
            "label_smooth": AOI_LABEL_SMOOTH,
            "use_focal": AOI_USE_FOCAL,
            "focal_gamma": AOI_FOCAL_GAMMA,
            "min_samples": AOI_MIN_SAMPLES
        },
        "note": "AOI-pruned training: removed raw classes 2 and 4; norm stats collected; best checkpoints saved for pretrain and AOI."
    }
    with open(MODEL_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved FINAL model to {MODEL_PTH}")
    print(f"Saved metadata to {MODEL_META}")
    print(f"Best PRETRAIN -> {best_pretrain_path if best_val>=0 else 'n/a'} | Best AOI -> {best_aoi_path if best_aoi>=0 else 'n/a'}")
    print(f"Final norm_mean={np.round(norm_mean,4).tolist()} | norm_std={np.round(norm_std,4).tolist()}")
    if BAND_SELECTION_9 is not None:
        print(f"Locked band order (source indices): {BAND_SELECTION_9}")
    elif BAND_SELECTION_8 is not None:
        print(f"Locked band order (source indices): {BAND_SELECTION_8}")

# # scratch_cnn_aoi_pruned.py
# # Train a crowns classifier with classes 2 and 4 REMOVED (0-indexed raw labels).
# # Includes: training-time norm stats + band order lock, optional focal loss,
# # val confusion-matrix dump, and a DEFAULT AOI fine-tuning stage.
#
# import os, glob, warnings, random, json, collections
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import rasterio
# from shapely.geometry import box
# from shapely.ops import transform as shp_transform
# import pyproj
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# import torchvision.models as models
#
# # ---------- Optional EPT handler ----------
# try:
#     from pyforestscan.handlers import read_lidar  # used when LAS_DIR is None
# except Exception:
#     read_lidar = None
#
# # ===============================
# # CONFIG (pretrain)
# # ===============================
# TRAINING_GPKG = "/Users/iosefa/repos/sam/trainings_filtered.gpkg"
# INDIVIDUAL_CROWNS_DIR = "/Users/iosefa/repos/sam/crowns_img"
# INDIVIDUAL_CROWNS_POINTS_DIR = "/Users/iosefa/repos/sam/crowns_las"   # if None, uses EPT
#
# # EPT fallback (used if INDIVIDUAL_CROWNS_POINTS_DIR is None or LAS for a key is missing)
# EPT_JSON = "/Users/iosefa/Downloads/ept6635/ept.json"
# EPT_SRS  = "EPSG:6635"   # CRS of the EPT dataset (must be correct)
#
# MODEL_PTH  = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft.pth"
# MODEL_META = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft.meta.json"
#
# # IMPORTANT: classes (raw, 0-index) to DROP from training (your AOI never has these)
# DROP_RAW_CLASSES = {2, 4}
#
# MAX_POINTS = 2048           # only downsample if > MAX_POINTS
# IMG_SIZE   = 320
# BATCH_SIZE = 16
# EPOCHS     = 30
# PATIENCE   = 6              # early stop on val acc
# WARMUP_FREEZE_EPOCHS = 2    # freeze backbone for first N epochs
# LR_BACKBONE = 3e-5
# LR_HEAD     = 8e-5
# WD         = 1e-4
# ADAMW_EPS  = 1e-8
# RANDOM_SEED= 42
# FREEZE_RESNET_INIT = True
# VAL_SPLIT  = 0.1
#
# # If your chips are 9- or 8-band stacks, pick 3 (WV3/WV2 examples below) — 0-based indices
# BAND_SELECTION_9 = [4, 2, 1]   # WV3 NIR, Red, Green
# BAND_SELECTION_8 = [6, 4, 2]   # WV2 NIR, Red, Green
#
# # Image robustness
# MIN_FINITE_PIX = 50
#
# # Regularization
# USE_MIXUP   = True
# MIXUP_ALPHA = 0.3
# LABEL_SMOOTH = 0.02
#
# # Loss choice
# USE_FOCAL = True   # toggle: True -> FocalLoss, False -> LabelSmoothingCE
# FOCAL_GAMMA = 1.5
#
# # ===============================
# # CONFIG (AOI fine-tune — DEFAULT: enabled)
# # ===============================
# AOI_GPKG               = "/Users/iosefa/repos/sam/overlap_labels.gpkg"  # same labels you use for priors/eval
# AOI_LABEL_COLUMN       = "class"   # 1-indexed raw labels in your GPKG
# AOI_VAL_SPLIT          = 0.0       # fine-tune is usually small; do full-train by default
# AOI_EPOCHS             = 10
# AOI_PATIENCE           = 3
# AOI_BATCH_SIZE         = 12
# AOI_LR_BACKBONE        = 1e-5
# AOI_LR_HEAD            = 3e-5
# AOI_WD                 = 5e-5
# AOI_USE_MIXUP          = False     # keep it off for fine-tune
# AOI_LABEL_SMOOTH       = 0.01
# AOI_USE_FOCAL          = True
# AOI_FOCAL_GAMMA        = 1.25
# AOI_ENABLE             = True      # default ON (will silently skip if GPKG missing)
# AOI_MIN_SAMPLES        = 20        # require at least this many samples post-prune to run
#
# # ===============================
# # Reproducibility
# # ===============================
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)
# torch.cuda.manual_seed_all(RANDOM_SEED)
#
# def dev():
#     if torch.backends.mps.is_available(): return torch.device("mps")
#     if torch.cuda.is_available():         return torch.device("cuda")
#     return torch.device("cpu")
#
# # ===============================
# # READ & MATCH TIF/LAS + LABELS (PRETRAIN)
# # ===============================
# gdf_labels = gpd.read_file(TRAINING_GPKG)
# # convert to raw 0-index labels in your file
# gdf_labels["feature_class"] = pd.to_numeric(gdf_labels["class"], errors="coerce").astype("Int64") - 1
# gdf_labels = gdf_labels.dropna(subset=["feature_class"])
#
# tif_files = sorted(glob.glob(os.path.join(INDIVIDUAL_CROWNS_DIR, "*.tif")))
# las_files = sorted(glob.glob(os.path.join(INDIVIDUAL_CROWNS_POINTS_DIR, "*.las"))) if INDIVIDUAL_CROWNS_POINTS_DIR else []
# las_all   = las_files
#
# tif_dict = {Path(p).stem: p for p in tif_files}
# las_dict = {Path(p).stem: p for p in las_all}
#
# keys_all = sorted(tif_dict.keys())
#
# records_all, geoms_all, labels_raw_all, keys_ok_all = [], [], [], []
#
# if gdf_labels.crs is None or gdf_labels.crs.is_geographic:
#     gdf_labels = gdf_labels.to_crs(6635)
#
# for key in keys_all:
#     tif_path = tif_dict[key]
#     las_path = las_dict.get(key, None)
#
#     with rasterio.open(tif_path) as src:
#         left, bottom, right, top = src.bounds
#         crs_tif = src.crs
#         if crs_tif is None:
#             warnings.warn(f"{tif_path} has no CRS; skipping.")
#             continue
#
#     chip_geom = box(left, bottom, right, top)
#     chip_gdf  = gpd.GeoDataFrame({"key":[key]}, geometry=[chip_geom], crs=crs_tif)
#     if chip_gdf.crs != gdf_labels.crs:
#         chip_gdf = chip_gdf.to_crs(gdf_labels.crs)
#         chip_geom = chip_gdf.geometry.iloc[0]
#
#     hits = gdf_labels[gdf_labels.geometry.intersects(chip_geom)]
#     if hits.empty:
#         continue
#
#     inter_areas = hits.geometry.intersection(chip_geom).area
#     idx = inter_areas.idxmax()
#     raw_label = hits.loc[idx, "feature_class"]
#     if pd.isna(raw_label):
#         continue
#
#     raw_label = int(raw_label)
#     # collect all, we will filter after
#     records_all.append((tif_path, las_path, raw_label, key, hits.loc[idx, "geometry"]))
#     geoms_all.append(hits.loc[idx, "geometry"])
#     labels_raw_all.append(raw_label)
#     keys_ok_all.append(key)
#
# print(f"Matched crowns with label & geometry: {len(records_all)} (before pruning)")
#
# # ---- PRUNE classes you don't want in AOI ----
# mask_keep = [y not in DROP_RAW_CLASSES for y in labels_raw_all]
# records = [rec for rec, keep in zip(records_all, mask_keep) if keep]
# geoms   = [g   for g,   keep in zip(geoms_all,   mask_keep) if keep]
# labels_raw = [y for y,   keep in zip(labels_raw_all, mask_keep) if keep]
# keys_ok = [k  for k,   keep in zip(keys_ok_all,  mask_keep) if keep]
#
# print(f"Kept after pruning classes {sorted(DROP_RAW_CLASSES)}: {len(records)}")
#
# gdf_records = gpd.GeoDataFrame({"key": keys_ok, "raw_label": labels_raw},
#                                geometry=geoms, crs=gdf_labels.crs)
#
# # Build contiguous class map for *remaining* raw labels
# RAW_LABELS_KEPT = sorted(set(labels_raw))
# CLASS_MAP = {lbl: i for i, lbl in enumerate(RAW_LABELS_KEPT)}  # raw->contiguous
# INV_CLASS_MAP = {i: lbl for lbl, i in CLASS_MAP.items()}
# NUM_CLASSES = len(CLASS_MAP)
#
# print(f"Raw labels kept: {RAW_LABELS_KEPT}  -> contiguous ids 0..{NUM_CLASSES-1}")
# print("Class histogram (raw kept):", dict(collections.Counter(labels_raw)))
# print("Class histogram (contiguous ids):", dict(collections.Counter([CLASS_MAP[y] for y in labels_raw])))
#
# # ===============================
# # TRAINING-TIME NORM STATS & BAND LOCK
# # ===============================
# RUN_STATS = {
#     "sum":   np.zeros(3, dtype=np.float64),
#     "sum2":  np.zeros(3, dtype=np.float64),
#     "count": np.zeros(3, dtype=np.int64),
# }
# EXPECTED_BAND_IDX = None
# if BAND_SELECTION_9 is not None:
#     EXPECTED_BAND_IDX = BAND_SELECTION_9
# elif BAND_SELECTION_8 is not None:
#     EXPECTED_BAND_IDX = BAND_SELECTION_8
#
# # ===============================
# # IO / PREPROC
# # ===============================
# def _apply_nodata_to_nan(arr, src):
#     arr = arr.astype(np.float32, copy=False)
#     if src.nodata is not None and np.isfinite(src.nodata):
#         arr[arr == src.nodata] = np.nan
#     return arr
#
# def _augment_image_cheap(img):
#     # img: (C,H,W) in [0,1] roughly after percentile norm
#     if random.random() < 0.5:
#         img = img[:, :, ::-1].copy()
#     if random.random() < 0.5:
#         img = img[:, ::-1, :].copy()
#     C = img.shape[0]
#     for c in range(C):
#         g = 1.0 + np.random.uniform(-0.15, 0.15)
#         b = np.random.uniform(-0.07, 0.07)
#         img[c] = np.clip(img[c]*g + b, 0.0, 1.0)
#     return img
#
# def read_image_as_3band_tensor(tif_path, training=False):
#     with rasterio.open(tif_path) as src:
#         img = src.read()                            # (C,H,W)
#         img = _apply_nodata_to_nan(img, src)
#
#     C = img.shape[0]
#     if C == 9 and BAND_SELECTION_9:
#         img = img[BAND_SELECTION_9, :, :]
#     elif C == 8 and BAND_SELECTION_8:
#         img = img[BAND_SELECTION_8, :, :]
#     elif C >= 3:
#         img = img[:3, :, :]
#     else:
#         raise ValueError(f"Unexpected band count {C} in {tif_path}")
#
#     img2 = []
#     for c in range(img.shape[0]):
#         band = img[c]
#         finite = np.isfinite(band)
#         if not np.any(finite):
#             img2.append(np.zeros_like(band, dtype=np.float32)); continue
#         lo, hi = np.nanpercentile(band[finite], [2, 98])
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
#             if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#                 lo, hi = 0.0, 1.0
#         band = np.clip(band, lo, hi)
#         band = (band - lo) / max(hi - lo, 1e-6)
#         img2.append(band.astype(np.float32))
#     img = np.stack(img2, axis=0)
#
#     if training:
#         img = _augment_image_cheap(img)
#
#     t = torch.from_numpy(img).unsqueeze(0)
#     t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
#
#     t[~torch.isfinite(t)] = 0.0
#
#     # -------- training-time running stats (post percentile norm) --------
#     if training:
#         with torch.no_grad():
#             for c in range(min(3, t.shape[0])):
#                 band_np = t[c].cpu().numpy()
#                 m = float(np.nanmean(band_np))
#                 v = float(np.nanvar(band_np))
#                 n = band_np.size
#                 RUN_STATS["sum"][c]  += m * n
#                 RUN_STATS["sum2"][c] += (v + m*m) * n   # E[x^2] * n
#                 RUN_STATS["count"][c]+= n
#
#     # -------- use persisted normalization if available (else ImageNet fallback) --------
#     norm_mean = [0.485, 0.456, 0.406]
#     norm_std  = [0.229, 0.224, 0.225]
#     try:
#         if os.path.exists(MODEL_META):
#             with open(MODEL_META, "r") as _f:
#                 _meta_prev = json.load(_f)
#                 if "norm_mean" in _meta_prev and "norm_std" in _meta_prev:
#                     norm_mean = _meta_prev["norm_mean"]
#                     norm_std  = _meta_prev["norm_std"]
#     except Exception:
#         pass
#
#     mean = torch.tensor(norm_mean, dtype=torch.float32).view(-1,1,1)
#     std  = torch.tensor(norm_std,  dtype=torch.float32).view(-1,1,1)
#     if mean.shape[0] != t.shape[0]:
#         mean = mean[:t.shape[0]]
#         std  = std[:t.shape[0]]
#
#     t = (t - mean) / (std.clamp_min(1e-6))
#     return t.float()
#
# def _pdal_struct_to_xyz(pc_obj):
#     arr = pc_obj[0] if isinstance(pc_obj, (list, tuple)) else pc_obj
#     if arr is None:
#         return np.zeros((0,3), dtype=np.float32)
#     if getattr(arr, "dtype", None) is None or arr.dtype.names is None:
#         a = np.asarray(arr)
#         return a[:, :3].astype(np.float32) if a.ndim == 2 and a.shape[1] >= 3 else np.zeros((0,3), dtype=np.float32)
#     names = [n.lower() for n in arr.dtype.names]
#     if all(n in names for n in ("x","y","z")):
#         X = arr[arr.dtype.names[names.index("x")]]
#         Y = arr[arr.dtype.names[names.index("y")]]
#         Z = arr[arr.dtype.names[names.index("z")]]
#     else:
#         cols = arr.dtype.names[:3]
#         X = arr[cols[0]]; Y = arr[cols[1]]; Z = arr[cols[2]]
#     return np.stack([np.asarray(X), np.asarray(Y), np.asarray(Z)], axis=1).astype(np.float32)
#
# def _reproject_points(xyz, src_crs, dst_crs):
#     if xyz.shape[0] == 0 or src_crs == dst_crs:
#         return xyz
#     fwd = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
#     xs, ys = fwd(xyz[:,0], xyz[:,1])
#     out = xyz.copy()
#     out[:,0] = xs
#     out[:,1] = ys
#     return out
#
# def read_points_for_training(las_path, crown_poly_in_label_crs, label_crs):
#     # A) LAS
#     if las_path is not None and Path(las_path).exists():
#         import laspy
#         with laspy.open(las_path) as f:
#             las = f.read()
#             pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
#         return pts
#     # B) EPT fallback
#     if read_lidar is None or not Path(EPT_JSON).exists():
#         return np.zeros((0,3), dtype=np.float32)
#     try:
#         label_crs = pyproj.CRS.from_user_input(label_crs)
#         ept_crs   = pyproj.CRS.from_user_input(EPT_SRS)
#     except Exception:
#         return np.zeros((0,3), dtype=np.float32)
#     if label_crs != ept_crs:
#         to_ept = pyproj.Transformer.from_crs(label_crs, ept_crs, always_xy=True).transform
#         poly_q = shp_transform(to_ept, crown_poly_in_label_crs)
#     else:
#         poly_q = crown_poly_in_label_crs
#     minx, miny, maxx, maxy = poly_q.bounds
#     bounds = (minx, miny, maxx, maxy)
#     try:
#         pc = read_lidar(EPT_JSON, EPT_SRS, bounds, crop_poly=True, poly=poly_q.wkt)
#         xyz = _pdal_struct_to_xyz(pc)
#         if xyz.shape[0] > 0 and (label_crs != ept_crs):
#             xyz = _reproject_points(xyz, ept_crs, label_crs)
#         return xyz.astype(np.float32)
#     except Exception:
#         return np.zeros((0,3), dtype=np.float32)
#
# def normalize_points_xy_center(points):
#     if points.size == 0:
#         return points
#     xy = points[:, :2]
#     z  = points[:, 2:3]
#     xy_c = xy - xy.mean(axis=0, keepdims=True)
#     std = xy_c.std(axis=0, keepdims=True)
#     std[std == 0] = 1e-6
#     xy_n = xy_c / std
#     z_std = z.std()
#     z_std = z_std if np.isfinite(z_std) and z_std > 0 else 1e-6
#     z_n = (z - z.mean()) / z_std
#     out = np.hstack([xy_n, z_n]).astype(np.float32)
#     out[~np.isfinite(out)] = 0.0
#     return out
#
# def downsample_if_needed(points, n=MAX_POINTS):
#     n_pts = points.shape[0]
#     if n_pts == 0:
#         return points
#     if n_pts > n:
#         idx = np.random.choice(n_pts, n, replace=False)
#         out = points[idx, :]
#     else:
#         out = points
#     out = out.astype(np.float32)
#     out[~np.isfinite(out)] = 0.0
#     return out
#
# # ===============================
# # DATASET
# # ===============================
# class CrownDataset(Dataset):
#     def __init__(self, recs, class_map, label_crs, training=False):
#         self.recs = recs
#         self.class_map = class_map
#         self.label_crs = label_crs
#         self.training = training
#
#     def __len__(self):
#         return len(self.recs)
#
#     def __getitem__(self, idx):
#         tif_path, las_path, y_raw, key, crown_geom = self.recs[idx]
#         y = self.class_map[y_raw]
#
#         # image
#         img = read_image_as_3band_tensor(tif_path, training=self.training)
#         img[~torch.isfinite(img)] = 0.0
#
#         # points
#         pts = read_points_for_training(las_path, crown_geom, self.label_crs)
#         pts = normalize_points_xy_center(pts)
#         pts = downsample_if_needed(pts, MAX_POINTS)  # keep variable length
#
#         # augs (training only)
#         if self.training and pts.shape[0] > 0:
#             # random small rotation of points
#             theta = np.random.uniform(0.0, 2.0*np.pi)
#             c, s = np.cos(theta), np.sin(theta)
#             xy = pts[:, :2].copy()
#             pts[:, 0] = c * xy[:, 0] - s * xy[:, 1]
#             pts[:, 1] = s * xy[:, 0] + c * xy[:, 1]
#
#             # random dropout & jitter
#             keep_ratio = np.random.uniform(0.7, 1.0)
#             n = pts.shape[0]
#             k = max(1, int(n * keep_ratio))
#             idx_keep = np.random.choice(n, k, replace=False)
#             pts = pts[idx_keep, :]
#             pts[:, 0:2] += np.random.normal(0.0, 0.02, size=(pts.shape[0], 2)).astype(np.float32)
#             pts[:, 2:3] += np.random.normal(0.0, 0.03, size=(pts.shape[0], 1)).astype(np.float32)
#
#             # flips aligned to image
#             if random.random() < 0.5:
#                 img = torch.flip(img, dims=[2])  # horizontal
#                 pts[:, 0] = -pts[:, 0]
#             if random.random() < 0.5:
#                 img = torch.flip(img, dims=[1])  # vertical
#                 pts[:, 1] = -pts[:, 1]
#
#         pts_t = torch.from_numpy(pts.astype(np.float32))
#         length = torch.tensor(pts_t.shape[0], dtype=torch.long)
#         return img, pts_t, length, y, key
#
# # ===============================
# # MODEL (keep head shapes compatible with classifier script)
# # ===============================
# class CombinedNet(nn.Module):
#     def __init__(self, num_classes, freeze_resnet=True):
#         super().__init__()
#         base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
#         if freeze_resnet:
#             for p in base.parameters():
#                 p.requires_grad = False
#         self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1) for RN34
#
#         self.fc1 = nn.Linear(3, 64, bias=True)
#         self.fc2 = nn.Linear(64, 128, bias=True)
#         self.fc3 = nn.Linear(128, 256, bias=True)  # time-dim features (T,256)
#
#         # Max pool -> 256-d vector; keep fc4 input=256 to match your classifier script
#         self.fc4 = nn.Linear(256, 128, bias=True)
#         self.fc5 = nn.Linear(128, 64, bias=True)
#         self.dropout = nn.Dropout(0.2)
#         self.classifier = nn.Linear(512 + 64, num_classes)  # 512 (img RN34) + 64 (pts)
#
#     def forward(self, img, pts, lengths=None):
#         x_img = self.backbone(img).view(img.size(0), -1)     # (B,512)
#
#         h = F.relu(self.fc1(pts))
#         h = F.relu(self.fc2(h))
#         h = F.relu(self.fc3(h))                               # (B,T,256)
#
#         if lengths is not None:
#             B, T, C = h.shape
#             ar = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
#             mask = ar >= lengths.unsqueeze(1)
#             h = h.masked_fill(mask.unsqueeze(-1), float("-inf"))
#
#         h, _ = torch.max(h, dim=1)                            # (B,256)
#         h[~torch.isfinite(h)] = 0.0
#         h = F.relu(self.fc4(h))
#         h = F.relu(self.fc5(h))                               # (B,64)
#
#         x = torch.cat([x_img, h], dim=1)                      # (B,576)
#         x = self.dropout(x)
#         return self.classifier(x)
#
# # ===============================
# # Collate (variable-length padding per batch)
# # ===============================
# def collate_variable(batch):
#     imgs, pts_list, lengths_list, ys, keys = zip(*batch)
#     imgs = torch.stack(imgs, dim=0)
#     lengths = torch.stack(lengths_list, dim=0)
#     Tmax = int(lengths.max().item()) if len(lengths_list) else 0
#     if Tmax == 0:
#         pts_padded = torch.zeros((len(batch), 1, 3), dtype=torch.float32)
#         lengths = torch.zeros((len(batch),), dtype=torch.long)
#     else:
#         pts_padded = torch.zeros((len(batch), Tmax, 3), dtype=torch.float32)
#         for i, pts in enumerate(pts_list):
#             n = pts.shape[0]
#             if n > 0:
#                 pts_padded[i, :n, :] = pts
#     ys = torch.tensor(ys, dtype=torch.long)
#     return imgs, pts_padded, lengths, ys, list(keys)
#
# # ===============================
# # Helpers
# # ===============================
# def dev_loader_opts():
#     if torch.backends.mps.is_available():
#         return torch.device("mps"), 0, False
#     if torch.cuda.is_available():
#         return torch.device("cuda"), 2, True
#     return torch.device("cpu"), 0, False
#
# class LabelSmoothingCE(nn.Module):
#     def __init__(self, weight=None, smoothing=0.0):
#         super().__init__()
#         self.weight = weight
#         self.smoothing = float(smoothing)
#     def forward(self, logits, target):
#         if self.smoothing <= 0.0:
#             return F.cross_entropy(logits, target, weight=self.weight)
#         num_classes = logits.size(-1)
#         logprobs = F.log_softmax(logits, dim=-1)
#         with torch.no_grad():
#             true_dist = torch.zeros_like(logprobs)
#             true_dist.fill_(self.smoothing / (num_classes - 1))
#             true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
#         loss = - (true_dist * logprobs)
#         if self.weight is not None:
#             loss = loss * self.weight.view(1, -1)
#         return loss.sum(dim=1).mean()
#
# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=1.5, smoothing=0.0):
#         super().__init__()
#         self.weight = weight
#         self.gamma = gamma
#         self.smoothing = smoothing
#     def forward(self, logits, target):
#         n_classes = logits.size(-1)
#         logp = F.log_softmax(logits, dim=-1)
#         with torch.no_grad():
#             true_dist = torch.zeros_like(logp)
#             true_dist.fill_(self.smoothing / (n_classes - 1))
#             true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
#         p = torch.exp(logp)
#         focal = (1 - p) ** self.gamma
#         loss = - focal * true_dist * logp
#         if self.weight is not None:
#             loss = loss * self.weight.view(1, -1)
#         return loss.sum(dim=1).mean()
#
# def maybe_mixup_logits(logits, y, alpha):
#     if alpha <= 0:
#         return logits, y, 1.0
#     lam = np.random.beta(alpha, alpha)
#     B = logits.size(0)
#     idx = torch.randperm(B, device=logits.device)
#     logits_mix = lam * logits + (1 - lam) * logits[idx]
#     return logits_mix, (y, y[idx]), lam
#
# def mixup_loss(logits_mix, y_pair, lam, criterion):
#     y1, y2 = y_pair
#     return lam * criterion(logits_mix, y1) + (1 - lam) * criterion(logits_mix, y2)
#
# @torch.no_grad()
# def eval_epoch(model, loader, criterion, device, num_classes):
#     model.eval()
#     loss_sum, correct, total = 0.0, 0, 0
#     cm = np.zeros((num_classes, num_classes), dtype=np.int64)
#     for img, pts, lengths, y, _ in loader:
#         img, pts, lengths = img.to(device), pts.to(device), lengths.to(device)
#         y = torch.as_tensor(y, dtype=torch.long, device=device)
#         img[~torch.isfinite(img)] = 0.0
#         pts[~torch.isfinite(pts)] = 0.0
#         logits = model(img, pts, lengths)
#         loss = criterion(logits, y)
#         loss_sum += loss.item() * y.size(0)
#         pred = logits.argmax(dim=1)
#         correct += (pred == y).sum().item()
#         total += y.size(0)
#         for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
#             cm[t, p] += 1
#     return loss_sum / max(total,1), correct / max(total,1), cm
#
# def run_epoch(model, loader, criterion, optimizer=None, device=torch.device("cpu"), use_mixup=True):
#     train = optimizer is not None
#     model.train() if train else model.eval()
#     loss_sum, correct, total = 0.0, 0, 0
#     for img, pts, lengths, y, keys in loader:
#         img, pts, lengths = img.to(device), pts.to(device), lengths.to(device)
#         y = torch.as_tensor(y, dtype=torch.long, device=device)
#         img[~torch.isfinite(img)] = 0.0
#         pts[~torch.isfinite(pts)] = 0.0
#
#         logits = model(img, pts, lengths)
#
#         if use_mixup and train:
#             logits_mix, y_pair, lam = maybe_mixup_logits(logits, y, MIXUP_ALPHA)
#             loss = mixup_loss(logits_mix, y_pair, lam, criterion)
#             pred = logits_mix.argmax(dim=1)
#         else:
#             loss = criterion(logits, y)
#             pred = logits.argmax(dim=1)
#
#         if train:
#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()
#
#         loss_sum += loss.item() * y.size(0)
#         correct += (pred == y).sum().item()
#         total += y.size(0)
#     return loss_sum / max(total,1), correct / max(total,1)
#
# def pretty_cm(cm):
#     with np.printoptions(linewidth=120):
#         return "\n" + "\n".join([" ".join([f"{v:5d}" for v in row]) for row in cm])
#
# # ===============================
# # AOI fine-tune helpers
# # ===============================
# def build_aoi_records(aoi_gpkg, label_col, dropped_raw_classes, tif_dict, las_dict, target_crs):
#     if not Path(aoi_gpkg).exists():
#         return []
#     g = gpd.read_file(aoi_gpkg)
#     if label_col not in g.columns:
#         return []
#     # to raw-0-index
#     g["feature_class"] = pd.to_numeric(g[label_col], errors="coerce").astype("Int64") - 1
#     g = g.dropna(subset=["feature_class"])
#     # align CRS
#     if g.crs is None:
#         try:
#             g = g.set_crs(target_crs)
#         except Exception:
#             pass
#     if g.crs != target_crs:
#         g = g.to_crs(target_crs)
#
#     # spatially match to chips by intersection and take max overlap (like pretrain)
#     recs = []
#     for key, tif_path in tif_dict.items():
#         with rasterio.open(tif_path) as src:
#             chip_geom = box(*src.bounds)
#             chip_gdf  = gpd.GeoDataFrame({"key":[key]}, geometry=[chip_geom], crs=src.crs)
#             if chip_gdf.crs != g.crs:
#                 chip_gdf = chip_gdf.to_crs(g.crs)
#                 chip_geom = chip_gdf.geometry.iloc[0]
#         hits = g[g.geometry.intersects(chip_geom)]
#         if hits.empty:
#             continue
#         inter_areas = hits.geometry.intersection(chip_geom).area
#         idx = inter_areas.idxmax()
#         raw_label = int(hits.loc[idx, "feature_class"])
#         if raw_label in dropped_raw_classes:
#             continue
#         las_path = las_dict.get(key, None)
#         recs.append((tif_dict[key], las_path, raw_label, key, hits.loc[idx, "geometry"]))
#     return recs
#
# def make_loader(recs, class_map, crs_str, batch_size, shuffle, sampler=None, workers=0, pin=False):
#     ds = CrownDataset(recs, class_map, crs_str, training=shuffle)
#     if sampler is not None:
#         loader = DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler,
#                             num_workers=workers, pin_memory=pin, collate_fn=collate_variable)
#     else:
#         loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
#                             num_workers=workers, pin_memory=pin, collate_fn=collate_variable)
#     return ds, loader
#
# # ===============================
# # MAIN
# # ===============================
# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method("spawn", force=True)
#     device, NUM_WORKERS, PIN_MEMORY = dev_loader_opts()
#     print(f"Using device: {device}. workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}")
#     if INDIVIDUAL_CROWNS_POINTS_DIR is None:
#         print(f"EPT fallback mode: reader={'ok' if read_lidar is not None else 'MISSING'}, "
#               f"ept_json_exists={Path(EPT_JSON).exists()}, ept_srs='{EPT_SRS}'")
#
#     # ----- stratified split after pruning (pretrain) -----
#     rng = np.random.RandomState(RANDOM_SEED)
#     if VAL_SPLIT > 0.0 and len(records) >= 10:
#         y_all = np.array([CLASS_MAP[y] for _,_,y,_,_ in records], dtype=np.int64)
#         val_idx, train_idx = [], []
#         for c in range(NUM_CLASSES):
#             cls_idx = np.where(y_all == c)[0]
#             rng.shuffle(cls_idx)
#             n_val_c = int(np.round(len(cls_idx) * VAL_SPLIT))
#             val_idx.extend(cls_idx[:n_val_c].tolist())
#             train_idx.extend(cls_idx[n_val_c:].tolist())
#         rng.shuffle(val_idx); rng.shuffle(train_idx)
#         train_recs = [records[i] for i in train_idx]
#         val_recs   = [records[i] for i in val_idx]
#     else:
#         train_recs = records
#         val_recs   = []
#
#     # ----- class weights & sampler on train only -----
#     y_train_ids = np.array([CLASS_MAP[y] for _,_,y,_,_ in train_recs], dtype=np.int64)
#     counts = np.bincount(y_train_ids, minlength=NUM_CLASSES).astype(np.float64)
#     inv_freq = 1.0 / np.clip(counts, 1, None)
#     ce_weights_np = inv_freq / (inv_freq.mean() + 1e-12)
#     ce_weight = torch.tensor(ce_weights_np, dtype=torch.float32, device=device)
#
#     # priors (for info only)
#     priors = (counts + 1.0) / (counts.sum() + NUM_CLASSES)  # Laplace
#     log_priors = np.log(priors + 1e-12)
#
#     print("Train class counts (pruned):", counts.tolist())
#     print("CE class weights:", ce_weights_np.tolist())
#     print("Log priors (pruned):", log_priors.tolist())
#
#     sample_weights = ce_weights_np[y_train_ids]
#     sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double),
#                                     num_samples=len(sample_weights),
#                                     replacement=True)
#
#     train_ds = CrownDataset(train_recs, CLASS_MAP, str(gdf_labels.crs), training=True)
#     val_ds   = CrownDataset(val_recs, CLASS_MAP, str(gdf_labels.crs), training=False) if val_recs else None
#
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
#                               sampler=sampler,
#                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
#                               collate_fn=collate_variable)
#     val_loader = (DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
#                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
#                              collate_fn=collate_variable)
#                   if val_ds else None)
#
#     print(f"Train size (pruned): {len(train_ds)} | Val size: {len(val_ds) if val_ds else 0}")
#
#     # Quick sanity: zero-point examples
#     zero_pts = 0
#     for _, (tif_path, las_path, y_raw, key, poly) in enumerate(train_recs[:100]):
#         pts = read_points_for_training(las_path, poly, str(gdf_labels.crs))
#         if pts.shape[0] == 0:
#             zero_pts += 1
#     print(f"Sampled train diagnostics: ~{zero_pts} / 100 examples had 0 LiDAR points.")
#
#     # ----- Model / Loss / Optim (pretrain) -----
#     model = CombinedNet(num_classes=NUM_CLASSES, freeze_resnet=FREEZE_RESNET_INIT).to(device)
#
#     # Initialize classifier bias with pruned log-priors (stabilizes early training)
#     with torch.no_grad():
#         model.classifier.bias.copy_(torch.tensor(log_priors, dtype=torch.float32, device=device))
#
#     if USE_FOCAL:
#         criterion = FocalLoss(weight=ce_weight, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTH)
#     else:
#         criterion = LabelSmoothingCE(weight=ce_weight, smoothing=LABEL_SMOOTH)
#
#     # Two LR groups: backbone (after unfreeze) and heads
#     head_params = list(model.fc1.parameters()) + list(model.fc2.parameters()) + \
#                   list(model.fc3.parameters()) + list(model.fc4.parameters()) + \
#                   list(model.fc5.parameters()) + list(model.classifier.parameters())
#     backbone_params = list(model.backbone.parameters())
#
#     optimizer = torch.optim.AdamW([
#         {"params": head_params, "lr": LR_HEAD, "weight_decay": WD},
#     ], eps=ADAMW_EPS)
#
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1))
#
#     best_val = -1.0
#     best_state = None
#     no_improve = 0
#
#     for epoch in range(1, EPOCHS+1):
#         # Unfreeze backbone after warmup
#         if epoch == WARMUP_FREEZE_EPOCHS + 1 and FREEZE_RESNET_INIT:
#             for p in model.backbone.parameters():
#                 p.requires_grad = True
#             optimizer.add_param_group({"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WD})
#             print(f"Epoch {epoch}: Unfroze backbone with LR={LR_BACKBONE:g}")
#
#         tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, use_mixup=USE_MIXUP)
#
#         if val_loader is not None and len(val_ds) > 0:
#             val_loss, val_acc, cm = eval_epoch(model, val_loader, criterion, device, NUM_CLASSES)
#             print(f"Epoch [{epoch:02d}/{EPOCHS}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
#                   f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\nConfusion matrix:\n{pretty_cm(cm)}")
#             if val_acc > best_val:
#                 best_val = val_acc
#                 best_state = {k: v.cpu() for k, v in model.state_dict().items()}
#                 no_improve = 0
#                 try:
#                     np.savetxt(Path(MODEL_PTH).with_suffix(".val_cm.csv"), cm, fmt="%d", delimiter=",")
#                 except Exception:
#                     pass
#             else:
#                 no_improve += 1
#                 if no_improve >= PATIENCE:
#                     print(f"Early stopping at epoch {epoch} (best Acc={best_val:.4f}).")
#                     break
#         else:
#             print(f"Epoch [{epoch:02d}/{EPOCHS}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f}")
#
#         scheduler.step()
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#         print(f"Loaded best val checkpoint (Acc={best_val:.4f}).")
#
#     # ===============================
#     # AOI FINE-TUNE (DEFAULT ON, skips gracefully if not available)
#     # ===============================
#     ran_aoi = False
#     if AOI_ENABLE and Path(AOI_GPKG).exists():
#         print("\n=== AOI fine-tune stage ===")
#         # Build AOI records by spatial match against same chip sources
#         aoi_recs = build_aoi_records(AOI_GPKG, AOI_LABEL_COLUMN, DROP_RAW_CLASSES,
#                                      tif_dict, las_dict, gdf_labels.crs)
#         if len(aoi_recs) >= AOI_MIN_SAMPLES:
#             # Check labels compatible with pretrain map
#             aoi_raw = [y for _,_,y,_,_ in aoi_recs]
#             aoi_kept = sorted(set(aoi_raw))
#             if any(lbl not in CLASS_MAP for lbl in aoi_kept):
#                 raise RuntimeError(f"AOI labels {aoi_kept} include classes not present in pretrain map {sorted(CLASS_MAP.keys())}")
#             # Use same CLASS_MAP so checkpoint stays compatible
#             y_ids = np.array([CLASS_MAP[y] for _,_,y,_,_ in aoi_recs], dtype=np.int64)
#             counts_aoi = np.bincount(y_ids, minlength=NUM_CLASSES).astype(np.float64)
#             inv_freq_aoi = 1.0 / np.clip(counts_aoi, 1, None)
#             ce_weights_aoi = inv_freq_aoi / (inv_freq_aoi.mean() + 1e-12)
#             ce_weight_aoi = torch.tensor(ce_weights_aoi, dtype=torch.float32, device=device)
#
#             print("AOI class counts (contiguous):", counts_aoi.tolist())
#             print("AOI CE weights:", ce_weights_aoi.tolist())
#
#             # Sampler (optional); for small sets, simple shuffle also works. We'll keep shuffle.
#             aoi_ds, aoi_loader = make_loader(aoi_recs, CLASS_MAP, str(gdf_labels.crs),
#                                              AOI_BATCH_SIZE, shuffle=True,
#                                              sampler=None, workers=NUM_WORKERS, pin=PIN_MEMORY)
#
#             # Optim/loss (low LR, all params trainable)
#             for p in model.backbone.parameters():
#                 p.requires_grad = True
#             params_all = [
#                 {"params": list(model.backbone.parameters()), "lr": AOI_LR_BACKBONE, "weight_decay": AOI_WD},
#                 {"params": list(model.fc1.parameters()) + list(model.fc2.parameters()) +
#                            list(model.fc3.parameters()) + list(model.fc4.parameters()) +
#                            list(model.fc5.parameters()) + list(model.classifier.parameters()),
#                  "lr": AOI_LR_HEAD, "weight_decay": AOI_WD},
#             ]
#             optimizer_aoi = torch.optim.AdamW(params_all, eps=ADAMW_EPS)
#
#             if AOI_USE_FOCAL:
#                 criterion_aoi = FocalLoss(weight=ce_weight_aoi, gamma=AOI_FOCAL_GAMMA, smoothing=AOI_LABEL_SMOOTH)
#             else:
#                 criterion_aoi = LabelSmoothingCE(weight=ce_weight_aoi, smoothing=AOI_LABEL_SMOOTH)
#
#             scheduler_aoi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_aoi, T_max=max(AOI_EPOCHS, 1))
#
#             best_aoi = -1.0
#             best_state_aoi = None
#             no_imp_aoi = 0
#
#             for epoch in range(1, AOI_EPOCHS+1):
#                 tr_loss, tr_acc = run_epoch(model, aoi_loader, criterion_aoi, optimizer_aoi,
#                                             device, use_mixup=AOI_USE_MIXUP)
#
#                 # optional tiny held-out split for AOI (off by default)
#                 if AOI_VAL_SPLIT > 0.0 and len(aoi_ds) >= 10:
#                     # quick split inside loop (AOI small; keep simple)
#                     n = len(aoi_recs)
#                     idx = np.arange(n)
#                     np.random.shuffle(idx)
#                     n_val = int(round(n * AOI_VAL_SPLIT))
#                     val_idx = idx[:n_val].tolist()
#                     tr_idx  = idx[n_val:].tolist()
#                     aoi_tr = [aoi_recs[i] for i in tr_idx]
#                     aoi_va = [aoi_recs[i] for i in val_idx]
#                     _, aoi_tr_loader = make_loader(aoi_tr, CLASS_MAP, str(gdf_labels.crs),
#                                                    AOI_BATCH_SIZE, shuffle=True, workers=NUM_WORKERS, pin=PIN_MEMORY)
#                     _, aoi_va_loader = make_loader(aoi_va, CLASS_MAP, str(gdf_labels.crs),
#                                                    AOI_BATCH_SIZE, shuffle=False, workers=NUM_WORKERS, pin=PIN_MEMORY)
#                     # one eval pass
#                     _, val_acc, cm_aoi = eval_epoch(model, aoi_va_loader, criterion_aoi, device, NUM_CLASSES)
#                 else:
#                     val_acc, cm_aoi = tr_acc, None
#
#                 print(f"[AOI {epoch:02d}/{AOI_EPOCHS}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f}" +
#                       (f" | Val Acc: {val_acc:.4f}" if cm_aoi is not None else ""))
#
#                 if val_acc > best_aoi:
#                     best_aoi = val_acc
#                     best_state_aoi = {k: v.cpu() for k, v in model.state_dict().items()}
#                     no_imp_aoi = 0
#                     if cm_aoi is not None:
#                         try:
#                             np.savetxt(Path(MODEL_PTH).with_suffix(".aoi_val_cm.csv"), cm_aoi, fmt="%d", delimiter=",")
#                         except Exception:
#                             pass
#                 else:
#                     no_imp_aoi += 1
#                     if no_imp_aoi >= AOI_PATIENCE:
#                         print(f"AOI early stopping at epoch {epoch} (best Acc={best_aoi:.4f}).")
#                         break
#
#                 scheduler_aoi.step()
#
#             if best_state_aoi is not None:
#                 model.load_state_dict(best_state_aoi)
#                 print(f"Loaded best AOI checkpoint (Acc={best_aoi:.4f}).")
#             ran_aoi = True
#         else:
#             print(f"AOI fine-tune skipped: found {len(aoi_recs)} samples (< {AOI_MIN_SAMPLES})")
#     else:
#         print("AOI fine-tune disabled or GPKG missing; skipping.")
#
#     # ----- Compute and persist final per-band norm stats (from RUN_STATS) -----
#     if RUN_STATS["count"].sum() > 0:
#         total = RUN_STATS["count"].astype(np.float64)
#         ex1   = RUN_STATS["sum"] / np.clip(total, 1, None)
#         ex2   = RUN_STATS["sum2"] / np.clip(total, 1, None)
#         var   = np.maximum(0.0, ex2 - ex1**2)
#         norm_mean = ex1.tolist()
#         norm_std  = np.sqrt(var).tolist()
#     else:
#         norm_mean = [0.485, 0.456, 0.406]
#         norm_std  = [0.229, 0.224, 0.225]
#
#     # ----- Save -----
#     torch.save(model.state_dict(), MODEL_PTH)
#     meta = {
#         "class_map": CLASS_MAP,                   # raw->contiguous (post-prune)
#         "inverse_class_map": INV_CLASS_MAP,       # contiguous->raw
#         "raw_labels_kept": RAW_LABELS_KEPT,       # e.g., [0,1,3,5]
#         "dropped_raw_labels": sorted(DROP_RAW_CLASSES),  # [2,4]
#         "num_classes": NUM_CLASSES,               # 4
#         "img_size": IMG_SIZE,
#         "max_points": MAX_POINTS,
#         "band_selection_9": BAND_SELECTION_9,
#         "band_selection_8": BAND_SELECTION_8,
#         # new fields consumed by the inference script:
#         "norm_mean": norm_mean,
#         "norm_std": norm_std,
#         "expected_band_idx": EXPECTED_BAND_IDX if EXPECTED_BAND_IDX is not None else None,
#         "freeze_resnet": False,
#         "architecture": "resnet34+pointMLP(maxpool)",
#         "weights": "IMAGENET1K_V1",
#         "class_weights": ce_weights_np.tolist(),
#         "val_split": VAL_SPLIT,
#         "random_seed": RANDOM_SEED,
#         "points_variable_length": True,
#         "log_priors": log_priors.tolist(),
#         "label_smoothing": LABEL_SMOOTH,
#         "use_focal": USE_FOCAL,
#         "focal_gamma": FOCAL_GAMMA,
#         "aoi_finetuned": ran_aoi,
#         "aoi_cfg": {
#             "aoi_gpkg": AOI_GPKG,
#             "epochs": AOI_EPOCHS,
#             "lr_backbone": AOI_LR_BACKBONE,
#             "lr_head": AOI_LR_HEAD,
#             "use_mixup": AOI_USE_MIXUP,
#             "label_smooth": AOI_LABEL_SMOOTH,
#             "use_focal": AOI_USE_FOCAL,
#             "focal_gamma": AOI_FOCAL_GAMMA,
#             "min_samples": AOI_MIN_SAMPLES
#         },
#         "note": "AOI-pruned training: removed raw classes 2 and 4; norm stats collected; default AOI fine-tune applied if available"
#     }
#     with open(MODEL_META, "w") as f:
#         json.dump(meta, f, indent=2)
#
#     print(f"Saved model to {MODEL_PTH}")
#     print(f"Saved metadata to {MODEL_META}")
#     print(f"Final norm_mean={np.round(norm_mean,4).tolist()} | norm_std={np.round(norm_std,4).tolist()}")
#     if EXPECTED_BAND_IDX is not None:
#         print(f"Locked band order (source indices): {EXPECTED_BAND_IDX}")