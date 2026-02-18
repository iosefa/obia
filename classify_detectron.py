import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pyforestscan.handlers import read_lidar
from tqdm import tqdm

# ===== metrics =====
try:
    from sklearn.metrics import classification_report, confusion_matrix
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

# =======================
# CONFIG — EDIT THESE
# =======================
PRED_CROWNS_GPKG  = "/Users/iosefa/repos/sam/cnn_predicted_ss1_overlap.gpkg"
PRED_CROWNS_LAYER = "crowns_predicted__crowns_pred"
BIG_WV3_TIF       = "/Users/iosefa/repos/sam/area51_subset1.tif"

# EPT source (STRICT; CRS must match crowns/raster)
EPT_JSON          = "/Users/iosefa/Downloads/ept6635/ept.json"
EPT_SRS           = "EPSG:6635"

# Classifier (4-class-pruned or 6-class)
# MODEL_PTH         = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft.pth"
# MODEL_META        = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft.meta.json"
MODEL_PTH         = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft1.best_aoi.pth"
MODEL_META        = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft1.meta.json"


# AOI labels (1-indexed 'class'); mapped to contiguous ids via meta["class_map"]
AOI_LABELS_GPKG   = "/Users/iosefa/repos/sam/overlap_labels.gpkg"
AOI_LAYER         = None

# Outputs
OUT_GPKG          = "/Users/iosefa/repos/sam/predicted_cnn_ss1.gpkg"
OUT_LAYER         = "predicted_cnn"
DIAG_CSV          = Path(OUT_GPKG).with_suffix(".diag.csv")
DEBUG_FULL_CSV    = Path(OUT_GPKG).with_suffix(".debug_full.csv")

# ===== evaluation outputs =====
EVAL_IOU_THR      = 0.5
EVAL_PREFIX       = Path(OUT_GPKG).with_suffix("")
EVAL_REPORT_TXT   = Path(str(EVAL_PREFIX) + ".cls_report.txt")
EVAL_CONFMAT_CSV  = Path(str(EVAL_PREFIX) + ".confusion_matrix.csv")
EVAL_MATCH_CSV    = Path(str(EVAL_PREFIX) + ".matches.csv")

# Image chip recovery/diagnostics
MIN_FINITE_PIX    = 50
BUFFER_METERS     = 0.5
FALLBACK_SIZE_M   = 2.0
REPORT_EVERY      = 25

# ====== reproduce training points behavior ======
PAD_TO_MAX        = False   # False = downsample only; True = pad to max_points

# ====== Calibration / priors / TTA / blending ======
USE_PRIOR_CORRECTION = True
PRIOR_MIX_ALPHA = 0.70       # match the earlier good baseline
PRIOR_SCALE     = 1.00
MAX_BIAS_ABS    = 1.20
TEMP            = 1.5        # earlier good setting

# Test-time augmentation
ENABLE_TTA            = True
TTA_IMG_FLIPS         = [None, "h", "v", "hv"]
TTA_PTS_ROT_K         = 8

# Branch blending (base; adapted per-sample)
W_IMG = 0.35
W_PTS = 0.65
IMG_SCALE   = 0.85
PTS_SCALE   = 1.00
LOGIT_CENTER = True

# ================= Anti-collapse controls (same as your good run) =================
CLASS3_ID = 3                 # contiguous id of the problematic class in pruned model
CLASS3_MARGIN_MIN   = 0.18
BRANCH_CONF_MIN     = 0.22
TTA_CLASS3_VOTE_MIN = 0.55
GLOBAL_SKEW_TRIGGER = 0.80
GLOBAL_NUDGE_STEP   = 0.10
GLOBAL_NUDGE_MAX    = 0.60
GLOBAL_WARMUP       = 20
STATIC_CLASS3_NUDGE = 0.00

# Do NOT lock to expected_band_idx; use per-stack selection like before
LOCK_BANDS_TO_META = False

# =======================
# RUNTIME DEVICE
# =======================
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

# =======================
# Helpers
# =======================
def entropy_np(p):
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())

def margin_top2(p):
    s = np.sort(p)
    return float(s[-1] - s[-2])

def adapt_weights(n_img, n_pts, base_img=0.30, base_pts=0.70):
    # Responsive tilt; pivot ~6, slope ~2
    s = 1.0 / (1.0 + np.exp(-((n_pts - 6.0) / 2.0)))
    w_pts = base_pts * 0.6 + 0.4 * s
    w_pts = float(np.clip(w_pts, 0.25, 0.85))
    w_img = 1.0 - w_pts
    return w_img, w_pts

# =======================
# MODEL
# =======================
class CombinedNet(nn.Module):
    def __init__(self, num_classes, freeze_resnet=False, resnet="resnet34", dual_pool=True, p_drop=0.2):
        super().__init__()
        if resnet == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            img_dim = 512
        else:
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            img_dim = 512
        if freeze_resnet:
            for p in base.parameters():
                p.requires_grad = False
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)

        self.fc1 = nn.Linear(3, 64, bias=True)
        self.fc2 = nn.Linear(64, 128, bias=True)
        self.fc3 = nn.Linear(128, 256, bias=True)

        self.dual_pool = dual_pool
        pts_in = 512 if dual_pool else 256
        self.fc4 = nn.Linear(pts_in, 128, bias=True)
        self.fc5 = nn.Linear(128, 64, bias=True)

        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(img_dim + 64, num_classes)

    def forward(self, img, pts, lengths=None, return_feats=False, branch_mask=None):
        x_img = self.backbone(img).view(img.size(0), -1)     # (B,512)

        h = F.relu(self.fc1(pts))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))                               # (B,T,256)

        if lengths is not None:
            B, T, C = h.shape
            ar = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
            mask = ar >= lengths.unsqueeze(1)
            h_max = h.masked_fill(mask.unsqueeze(-1), float("-inf")).max(dim=1).values
            if self.dual_pool:
                h_sum = h.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1)
                denom = lengths.clamp_min(1).unsqueeze(1).to(h.dtype)
                h_mean = h_sum / denom
                h_max[~torch.isfinite(h_max)] = 0.0
                h = torch.cat([h_max, h_mean], dim=1)        # (B,512)
            else:
                h_max[~torch.isfinite(h_max)] = 0.0
                h = h_max                                    # (B,256)
        else:
            if self.dual_pool:
                h = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=1)
            else:
                h = h.max(dim=1).values

        h = F.relu(self.fc4(h))
        h = self.dropout(h)
        h = F.relu(self.fc5(h))                               # (B,64)

        if branch_mask is not None:
            if not branch_mask.get("img", True):
                x_img = torch.zeros_like(x_img)
            if not branch_mask.get("pts", True):
                h = torch.zeros_like(h)

        x = torch.cat([x_img, h], dim=1)                      # (B,576)
        logits = self.classifier(x)
        if return_feats:
            return logits, x_img.norm(dim=1), h.norm(dim=1)
        return logits

# =======================
# IMAGE HELPERS
# =======================
def robust_percentile_norm(band):
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(band[finite], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
    band = np.clip(band, lo, hi)
    band = (band - lo) / max(hi - lo, 1e-6)
    return band.astype(np.float32)

def _read_by_polygon(src, poly):
    if not poly.is_valid:
        poly = poly.buffer(0)
    out, _ = rio_mask(src, [mapping(poly)], crop=True, filled=False, pad=True)
    return np.where(out.mask, np.nan, out.data).astype(np.float32)

def _read_by_buffered_polygon(src, poly, buffer_m):
    return _read_by_polygon(src, poly.buffer(buffer_m))

def _read_by_centroid_window(src, poly, win_size_m):
    cx, cy = poly.centroid.x, poly.centroid.y
    half = win_size_m / 2.0
    minx, miny, maxx, maxy = cx - half, cy - half, cx + half, cy + half
    win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    arr = src.read(window=win, boundless=True, fill_value=np.nan).astype(np.float32)
    if src.nodata is not None and np.isfinite(src.nodata):
        arr[arr == src.nodata] = np.nan
    return arr

def _select_bands_from_meta(data_chw, meta):
    C = data_chw.shape[0]
    if C == 9 and meta.get("band_selection_9") is not None:
        idx = list(meta["band_selection_9"])
    elif C == 8 and meta.get("band_selection_8") is not None:
        idx = list(meta["band_selection_8"])
    else:
        idx = list(range(min(3, C)))
    if max(idx) >= C:
        raise ValueError(f"Requested bands {idx} not present in raster with {C} bands.")
    return data_chw[idx, :, :], idx

def make_crown_tensor(src, poly, meta_img_size, meta):
    stats = {"stage":"poly", "finite_sum":0, "finite_per_band":[0,0,0],
             "std_per_band":[np.nan,np.nan,np.nan], "err":"", "bands_idx":[]}

    data = _read_by_polygon(src, poly)
    if not np.isfinite(data).any():
        stats["stage"] = "buffer_poly"
        data = _read_by_buffered_polygon(src, poly, BUFFER_METERS)
    if not np.isfinite(data).any():
        stats["stage"] = "centroid_win"
        data = _read_by_centroid_window(src, poly, FALLBACK_SIZE_M)

    arr, used_idx = _select_bands_from_meta(data, meta)
    stats["bands_idx"] = used_idx

    fin = [int(np.isfinite(arr[k]).sum()) for k in range(arr.shape[0])]
    stats["finite_per_band"] = fin
    stats["finite_sum"] = int(sum(fin))
    stats["std_per_band"] = [float(np.nanstd(arr[k])) for k in range(arr.shape[0])]

    if stats["finite_sum"] < MIN_FINITE_PIX and stats["stage"] != "centroid_win":
        stats["stage"] += "->centroid_win"
        data2 = _read_by_centroid_window(src, poly, FALLBACK_SIZE_M)
        arr2, used_idx2 = _select_bands_from_meta(data2, meta)
        fin2 = [int(np.isfinite(arr2[k]).sum()) for k in range(arr2.shape[0])]
        if sum(fin2) > stats["finite_sum"]:
            arr = arr2
            stats["bands_idx"] = used_idx2
            stats["finite_per_band"] = fin2
            stats["finite_sum"] = int(sum(fin2))
            stats["std_per_band"] = [float(np.nanstd(arr2[k])) for k in range(arr.shape[0])]

    norm = np.stack([robust_percentile_norm(arr[c]) for c in range(arr.shape[0])], axis=0)
    t = torch.from_numpy(norm).unsqueeze(0)
    t = F.interpolate(t, size=(meta_img_size, meta_img_size), mode="bilinear", align_corners=False).squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    t = ((t - mean) / std).float()
    return t, stats

# =======================
# POINTS HELPERS
# =======================
def _pdal_struct_to_xyz(pc_obj):
    arr = pc_obj[0] if isinstance(pc_obj, (list, tuple)) else pc_obj
    if arr is None:
        raise RuntimeError("read_lidar returned None.")
    if getattr(arr, "dtype", None) is not None and arr.dtype.names is not None:
        names_lower = [n.lower() for n in arr.dtype.names]
        if all(n in names_lower for n in ("x", "y", "z")):
            X = arr[arr.dtype.names[names_lower.index("x")]]
            Y = arr[arr.dtype.names[names_lower.index("y")]]
            Z = arr[arr.dtype.names[names_lower.index("z")]]
        else:
            f0, f1, f2 = arr.dtype.names[:3]
            X = arr[f0]; Y = arr[f1]; Z = arr[f2]
        xyz = np.stack([np.asarray(X), np.asarray(Y), np.asarray(Z)], axis=1).astype(np.float32)
    else:
        a = np.asarray(arr)
        if a.ndim != 2 or a.shape[1] < 3:
            raise RuntimeError(f"Unexpected point array shape from read_lidar: {a.shape}")
        xyz = a[:, :3].astype(np.float32)
    return xyz

def fetch_points_for_polygon(poly, ept_json, ept_srs):
    minx, miny, maxx, maxy = poly.bounds
    bounds = ([minx, maxx], [miny, maxy])
    pc = read_lidar(ept_json, ept_srs, bounds, crop_poly=True, poly=poly.wkt)
    pts = _pdal_struct_to_xyz(pc)
    if pts.shape[0] == 0:
        raise RuntimeError(f"EPT returned 0 points for bounds={bounds}")
    return pts

def normalize_points_xy_center(points):
    xy = points[:, :2]; z = points[:, 2:3]
    xy_c = xy - xy.mean(axis=0, keepdims=True)
    std = xy_c.std(axis=0, keepdims=True); std[std == 0] = 1e-6
    xy_n = xy_c / std
    z_std = z.std(); z_std = z_std if np.isfinite(z_std) and z_std > 0 else 1e-6
    z_n = (z - z.mean()) / z_std
    out = np.hstack([xy_n, z_n]).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite values in normalized point set.")
    return out

def downsample_if_needed(points, n_cap):
    if points.shape[0] > n_cap:
        idx = np.random.choice(points.shape[0], n_cap, replace=False)
        return points[idx, :].astype(np.float32)
    return points.astype(np.float32)

def pad_to_max(points, n_max):
    if points.shape[0] >= n_max:
        idx = np.random.choice(points.shape[0], n_max, replace=False)
        return points[idx, :].astype(np.float32), n_max
    add = np.random.choice(points.shape[0], n_max - points.shape[0], replace=True)
    out = np.vstack([points, points[add, :]]).astype(np.float32)
    return out, n_max

# ---- TTA helpers ----
def apply_img_flip(t_b1):
    outs = []
    for f in TTA_IMG_FLIPS:
        if f is None:
            outs.append(t_b1)
        elif f == "h":
            outs.append(torch.flip(t_b1, dims=[3]))
        elif f == "v":
            outs.append(torch.flip(t_b1, dims=[2]))
        elif f == "hv":
            outs.append(torch.flip(torch.flip(t_b1, dims=[2]), dims=[3]))
    return outs

def rotate_points_z(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    out = pts.copy()
    out[:, :2] = pts[:, :2] @ R.T
    return out

# =======================
# AOI priors (1-indexed -> contiguous ids)
# =======================
def expected_priors_from_gpkg(gpkg_path, layer_name, class_map):
    if not Path(gpkg_path).exists():
        return None
    g = gpd.read_file(gpkg_path, layer=layer_name) if layer_name else gpd.read_file(gpkg_path)
    if "class" not in g.columns:
        return None
    labels_raw0 = pd.to_numeric(g["class"], errors="coerce").dropna().astype(int) - 1  # 1-index -> 0-index raw
    raw2contig = {int(k): int(v) for k, v in meta["class_map"].items()} if isinstance(meta["class_map"], dict) else meta["class_map"]
    contig_ids = [raw2contig[r] for r in labels_raw0 if r in raw2contig]
    if not contig_ids:
        return None
    num_classes = max(raw2contig.values()) + 1
    counts = np.bincount(np.array(contig_ids, dtype=np.int64), minlength=num_classes).astype(np.float64)
    priors = counts / np.clip(counts.sum(), 1e-12, None)
    print(f"AOI prior counts (contiguous ids 0..{num_classes-1}): {counts.tolist()}")
    print(f"AOI priors: {np.round(priors, 4).tolist()}")
    return priors

# ===== evaluation helpers =====
def _ensure_int_series(s):
    return pd.to_numeric(s, errors="coerce").dropna().astype(int)

def _eval_match_and_report(pred_gdf, labels_gdf, inv_map, iou_thr=0.5):
    if "predicted_class" not in pred_gdf.columns:
        raise RuntimeError("pred_gdf missing 'predicted_class'")
    if "class" not in labels_gdf.columns:
        raise RuntimeError("labels_gdf missing 'class'")
    labels_gdf = labels_gdf.copy()
    labels_gdf["ref_raw"] = _ensure_int_series(labels_gdf["class"]) - 1

    if "key" in pred_gdf.columns and "key" in labels_gdf.columns:
        merged = pred_gdf[["key", "predicted_class", "geometry"]].merge(
            labels_gdf[["key", "ref_raw", "geometry"]], on="key", how="inner", suffixes=("_pred","_ref")
        )
        merged = gpd.GeoDataFrame(merged, geometry="geometry_pred", crs=pred_gdf.crs).rename(columns={"geometry_ref":"geom_ref"})
        merged["geom_ref"] = merged["geom_ref"]
        inter_area = merged["geometry_pred"].intersection(merged["geom_ref"]).area
        union_area = merged["geometry_pred"].union(merged["geom_ref"]).area
        iou = np.divide(inter_area, np.clip(union_area, 1e-12, None))
        merged["iou"] = iou
        pairs = merged[["key", "predicted_class", "ref_raw", "iou"]].rename(columns={"predicted_class":"pred_raw"})
        pairs = pairs[pairs["iou"] >= iou_thr].reset_index(drop=True)
    else:
        pred = pred_gdf.copy()
        lbl  = labels_gdf.copy()
        if pred.crs != lbl.crs:
            lbl = lbl.to_crs(pred.crs)
        pred = pred.reset_index(drop=True).reset_index(names="pid")
        lbl  = lbl.reset_index(drop=True).reset_index(names="vid")
        cand = gpd.sjoin(pred[["pid","geometry"]], lbl[["vid","geometry"]], how="inner", predicate="intersects").drop(columns=["index_right"])
        if len(cand) == 0:
            raise RuntimeError("No overlapping geometries found for evaluation.")
        pred_sub = pred.loc[cand["pid"], ["pid","geometry"]].rename(columns={"geometry":"geom_pred"})
        lbl_sub  = lbl.loc[cand["vid"], ["vid","geometry"]].rename(columns={"geometry":"geom_ref"})
        merged = cand.merge(pred_sub, on="pid").merge(lbl_sub, on="vid")
        inter = merged.apply(lambda r: r["geom_pred"].intersection(r["geom_ref"]), axis=1)
        inter_area = inter.area
        union_area = (merged["geom_pred"].area + merged["geom_ref"].area - inter_area)
        merged["iou"] = np.divide(inter_area, np.clip(union_area, 1e-12, None))
        merged = merged.sort_values(["pid","iou"], ascending=[True, False]).groupby("pid", as_index=False).first()
        keep = merged[merged["iou"] >= iou_thr]
        pairs = keep.merge(pred[["pid","predicted_class"]], on="pid").merge(lbl[["vid","ref_raw"]], on="vid")
        pairs = pairs.rename(columns={"predicted_class":"pred_raw"})[["pid","vid","iou","pred_raw","ref_raw"]]
        pairs = pairs.rename(columns={"pid":"key"})
    if len(pairs) == 0:
        raise RuntimeError(f"No matches at IoU ≥ {iou_thr}. Cannot produce classification report.")
    if not _HAVE_SK:
        raise RuntimeError("scikit-learn not installed. `pip install scikit-learn` to get classification reports.")
    y_true = pairs["ref_raw"].to_numpy()
    y_pred = pairs["pred_raw"].to_numpy()
    labels_all = np.unique(np.concatenate([y_true, y_pred]))
    report_str = classification_report(y_true, y_pred, labels=labels_all, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    cm_df = pd.DataFrame(cm, index=[f"ref_{i}" for i in labels_all], columns=[f"pred_{i}" for i in labels_all])
    return pairs, report_str, cm_df

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    device = get_device()
    print(f"Device: {device}")

    # Load meta
    with open(MODEL_META, "r") as f:
        meta = json.load(f)
    num_classes = int(meta["num_classes"])
    class_map   = {int(k): int(v) for k, v in meta["class_map"].items()} if isinstance(meta["class_map"], dict) else meta["class_map"]
    inv_map     = {v:k for k,v in class_map.items()}  # contiguous -> RAW
    img_size    = int(meta["img_size"])
    max_points  = int(meta.get("max_points", 2048))

    # Inspect checkpoint
    state = torch.load(MODEL_PTH, map_location="cpu")
    dual_pool = True
    if "fc4.weight" in state:
        dual_pool = (state["fc4.weight"].shape[1] == 512)
    arch = str(meta.get("architecture", "")).lower()
    resnet_kind = "resnet34" if "resnet34" in arch else ("resnet18" if "resnet18" in arch else "resnet34")

    if meta.get("expected_band_idx") is not None:
        print(f"Locked band order (source indices): {meta['expected_band_idx']}")  # informational only

    # Build and load
    model = CombinedNet(num_classes=num_classes,
                        freeze_resnet=meta.get("freeze_resnet", False),
                        resnet=resnet_kind,
                        dual_pool=dual_pool).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded model with dual_pool={dual_pool} (fc4 in_features = {'512' if dual_pool else '256'}), backbone={resnet_kind}")

    # ---- Priors (tempered) ----
    train_log_priors = np.array(meta.get("log_priors", [0.0]*num_classes), dtype=np.float32)
    train_priors = np.exp(train_log_priors); train_priors /= np.clip(train_priors.sum(), 1e-12, None)

    aoi_priors = expected_priors_from_gpkg(AOI_LABELS_GPKG, AOI_LAYER, class_map)
    if USE_PRIOR_CORRECTION and aoi_priors is not None:
        p_eff = (1.0 - PRIOR_MIX_ALPHA) * train_priors + PRIOR_MIX_ALPHA * aoi_priors
        p_eff = p_eff / np.clip(p_eff.sum(), 1e-12, None)
        raw_bias = np.log(np.clip(p_eff, 1e-6, 1.0)) - np.log(np.clip(train_priors, 1e-6, 1.0))
        scaled_bias = PRIOR_SCALE * raw_bias
        scaled_bias = np.clip(scaled_bias, -MAX_BIAS_ABS, MAX_BIAS_ABS)
        prior_bias_corr = torch.tensor(-scaled_bias, dtype=torch.float32, device=device)  # subtract later
        print("Using tempered prior correction:")
        print(f"  train_priors   = {np.round(train_priors,4).tolist()}")
        print(f"  aoi_priors     = {np.round(aoi_priors,4).tolist()}")
        print(f"  mixed_priors   = {np.round(p_eff,4).tolist()}")
        print(f"  raw_bias       = {np.round(raw_bias,3).tolist()}")
        print(f"  scaled&clipped = {np.round(scaled_bias,3).tolist()} | TEMP={TEMP}")
    else:
        prior_bias_corr = torch.zeros(num_classes, dtype=torch.float32, device=device)
        print(f"Prior correction disabled or AOI priors missing. TEMP={TEMP}")

    # Optional static nudge vs class-3
    static_corr = torch.zeros(num_classes, dtype=torch.float32, device=device)
    if STATIC_CLASS3_NUDGE > 0 and CLASS3_ID < num_classes:
        static_corr[CLASS3_ID] = float(STATIC_CLASS3_NUDGE)

    # Global skew brake state
    global_counts = np.zeros(num_classes, dtype=np.int64)
    global_nudge  = np.zeros(num_classes, dtype=np.float32)  # in logit units

    # Load predicted crowns
    crowns = gpd.read_file(PRED_CROWNS_GPKG, layer=PRED_CROWNS_LAYER)
    if crowns.crs is None:
        raise RuntimeError("Predicted crowns GPKG has no CRS.")
    print(f"Loaded {len(crowns)} predicted crowns.")

    if not Path(EPT_JSON).exists():
        raise RuntimeError(f"EPT JSON not found: {EPT_JSON}")
    if str(crowns.crs) != EPT_SRS:
        raise RuntimeError(f"CRS mismatch. crowns.crs='{crowns.crs}' but EPT_SRS='{EPT_SRS}'.")
    print(f"EPT mode OK. ept_json='{EPT_JSON}', srs='{EPT_SRS}'")

    if "key" not in crowns.columns:
        crowns["key"] = [f"crown_{i}" for i in range(len(crowns))]

    keys, pred_ids, prob_rows = [], [], []
    diags = []
    feat_img_norms, feat_pts_norms, sample_probs = [], [], []
    logits_sum = np.zeros(num_classes, dtype=np.float64)
    logits_sum_img = np.zeros(num_classes, dtype=np.float64)
    logits_sum_pts = np.zeros(num_classes, dtype=np.float64)
    entropies = []
    first_print_done = False

    with rasterio.open(BIG_WV3_TIF) as src:
        if src.crs is None:
            raise RuntimeError("Raster has no CRS.")
        if str(src.crs) != EPT_SRS:
            raise RuntimeError(f"Raster CRS '{src.crs}' must equal EPT_SRS '{EPT_SRS}'.")
        if src.crs != crowns.crs:
            raise RuntimeError("Raster CRS and crowns CRS differ (no reprojection).")

        for i, row in tqdm(crowns.iterrows(), total=len(crowns), desc="Classifying crowns"):
            key  = row["key"]
            geom = row.geometry
            if geom is None or geom.is_empty:
                raise RuntimeError(f"Empty geometry for key={key}")

            # IMAGE
            img_t, stats = make_crown_tensor(src, geom, img_size, meta)
            if stats["finite_sum"] == 0:
                raise RuntimeError(f"No finite pixels for key={key}")

            if not first_print_done:
                print(f"Using bands (0-based) from meta: {stats['bands_idx']}")
                first_print_done = True

            # POINTS
            pts = fetch_points_for_polygon(geom, EPT_JSON, EPT_SRS)
            pts = normalize_points_xy_center(pts)
            if PAD_TO_MAX:
                pts, eff_len = pad_to_max(pts, max_points)
            else:
                pts = downsample_if_needed(pts, max_points)
                eff_len = pts.shape[0]

            img_b1 = img_t.unsqueeze(0).to(device)
            pts_np = pts.astype(np.float32)

            # ---- helper: one forward producing blended & per-branch calibrated logits
            def blend_logits(li, lp, w_img_dyn, w_pts_dyn):
                li = IMG_SCALE * li
                lp = PTS_SCALE * lp
                lb = w_img_dyn * li + w_pts_dyn * lp
                if LOGIT_CENTER:
                    lb = lb - lb.mean(dim=1, keepdim=True)
                corr = prior_bias_corr + static_corr + torch.from_numpy(global_nudge).to(lb.device)
                lb = (lb - corr) / float(TEMP)
                li = (li - corr) / float(TEMP)
                lp = (lp - corr) / float(TEMP)
                return lb, li, lp

            # base view (for dynamic weights + diagnostics)
            length_t0 = torch.tensor([pts_np.shape[0]], dtype=torch.long, device=device)
            pts_t0    = torch.from_numpy(pts_np).unsqueeze(0).to(device)
            with torch.no_grad():
                lg_img0 = model(img_b1, pts_t0, lengths=length_t0, branch_mask={"img": True,  "pts": False})
                lg_pts0 = model(img_b1, pts_t0, lengths=length_t0, branch_mask={"img": False, "pts": True})
                _, n_img_t0, n_pts_t0 = model(img_b1, pts_t0, lengths=length_t0, return_feats=True)
            n_img = float(n_img_t0.item()); n_pts = float(n_pts_t0.item())
            w_img_dyn, w_pts_dyn = adapt_weights(n_img, n_pts, base_img=W_IMG, base_pts=W_PTS)

            # TTA views
            img_views = apply_img_flip(img_b1) if ENABLE_TTA else [img_b1]
            angs = [2*np.pi*k/float(TTA_PTS_ROT_K) for k in range(TTA_PTS_ROT_K)] if (ENABLE_TTA and TTA_PTS_ROT_K and TTA_PTS_ROT_K > 1) else [0.0]

            logits_accum = []
            tta_preds = []
            with torch.no_grad():
                for im in img_views:
                    for th in angs:
                        try:
                            pts_aug = rotate_points_z(pts_np, th).astype(np.float32)
                            if pts_aug.shape[0] == 0 or not np.isfinite(pts_aug).all():
                                continue
                            lt = torch.tensor([pts_aug.shape[0]], dtype=torch.long, device=device)
                            pt = torch.from_numpy(pts_aug).unsqueeze(0).to(device)
                            li = model(im, pt, lengths=lt, branch_mask={"img": True,  "pts": False})
                            lp = model(im, pt, lengths=lt, branch_mask={"img": False, "pts": True})
                            lb, _, _ = blend_logits(li, lp, w_img_dyn, w_pts_dyn)
                            if lb is not None and torch.isfinite(lb).all():
                                logits_accum.append(lb)
                                tta_preds.append(int(lb.argmax(dim=1).item()))
                        except Exception:
                            continue

            # fallback if TTA produced nothing
            if len(logits_accum) == 0:
                lb_fallback, _, _ = blend_logits(lg_img0, lg_pts0, w_img_dyn, w_pts_dyn)
                logits_accum = [lb_fallback]
                tta_preds = [int(lb_fallback.argmax(dim=1).item())]

            logits = torch.stack(logits_accum, dim=0).mean(dim=0)   # (1,C)
            probs_blend = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_blend  = int(probs_blend.argmax())
            m_blend     = margin_top2(probs_blend)

            # per-branch calibrated on base view (for guard/diagnostics)
            lb0, li0, lp0 = blend_logits(lg_img0, lg_pts0, w_img_dyn, w_pts_dyn)
            probs_img = F.softmax(li0, dim=1).cpu().numpy()[0]
            probs_pts = F.softmax(lp0, dim=1).cpu().numpy()[0]
            pred_img = int(probs_img.argmax()); m_img = margin_top2(probs_img)
            pred_pts = int(probs_pts.argmax()); m_pts = margin_top2(probs_pts)

            # --------- class-3 TTA vote gate + margin guard ----------
            pred = pred_blend
            if pred_blend == CLASS3_ID:
                vote_share_3 = (np.array(tta_preds) == CLASS3_ID).mean() if len(tta_preds) else 1.0
                veto_vote = vote_share_3 < TTA_CLASS3_VOTE_MIN
                veto_margin = m_blend < CLASS3_MARGIN_MIN

                if veto_vote or veto_margin:
                    logits_np = logits.cpu().numpy()[0].copy()
                    logits_np[CLASS3_ID] = -1e9
                    cand1 = int(logits_np.argmax())

                    cand2, cand2_margin = None, -1.0
                    if pred_img != CLASS3_ID and m_img > BRANCH_CONF_MIN:
                        cand2, cand2_margin = pred_img, m_img
                    if pred_pts != CLASS3_ID and m_pts > BRANCH_CONF_MIN:
                        if m_pts > cand2_margin:
                            cand2, cand2_margin = pred_pts, m_pts

                    if cand2 is not None:
                        pred = cand2
                    else:
                        pred = cand1

            # --------- global skew brake update ----------
            global_counts[pred] += 1
            seen = int(global_counts.sum())
            if seen >= GLOBAL_WARMUP:
                frac = global_counts / np.clip(seen, 1, None)
                j = int(frac.argmax())
                if frac[j] >= GLOBAL_SKEW_TRIGGER and global_nudge[j] < GLOBAL_NUDGE_MAX:
                    global_nudge[j] = float(min(GLOBAL_NUDGE_MAX, global_nudge[j] + GLOBAL_NUDGE_STEP))
                    print(f"[Skew brake] After {seen} preds, class {j} at {frac[j]:.2f} -> increase global_nudge[{j}] to {global_nudge[j]:.2f}")

            # diags
            feat_img_norms.append(n_img)
            feat_pts_norms.append(n_pts)
            entropies.append(entropy_np(probs_blend))
            logits_sum += logits.cpu().numpy()[0]
            logits_sum_img += li0.detach().cpu().numpy()[0]
            logits_sum_pts += lp0.detach().cpu().numpy()[0]

            keys.append(key)
            pred_ids.append(pred)
            prob_rows.append(probs_blend.tolist())
            sample_probs.append(float(probs_blend[pred]))

            diags.append({
                "key": key,
                "stage": stats["stage"],
                "finite_sum": stats["finite_sum"],
                "finite_b0": stats["finite_per_band"][0],
                "finite_b1": stats["finite_per_band"][1],
                "finite_b2": stats["finite_per_band"][2],
                "std_b0": stats["std_per_band"][0],
                "std_b1": stats["std_per_band"][1],
                "std_b2": stats["std_per_band"][2],
                "err": "",
                "pts_mode": "ept_poly",
                "ept_bounds": str(([geom.bounds[0], geom.bounds[2]], [geom.bounds[1], geom.bounds[3]])),
                "pts_n_after_clip": int(eff_len),
                "feat_norm_img": n_img,
                "feat_norm_pts": n_pts,
                "pred_id": pred,
                "pred_prob": float(probs_blend[pred]),
                "entropy": entropies[-1],
                "w_img_dyn": w_img_dyn, "w_pts_dyn": w_pts_dyn,
                "m_blend": m_blend, "m_img": m_img, "m_pts": m_pts,
                "pred_blend": pred_blend, "pred_img": pred_img, "pred_pts": pred_pts,
                "tta_vote_share_class3": (np.array(tta_preds) == CLASS3_ID).mean() if len(tta_preds) else 1.0,
                "global_nudge": global_nudge.tolist(),
            })

            if (i+1) % REPORT_EVERY == 0:
                print(f"[{i+1}/{len(crowns)}] stage={stats['stage']} fin={stats['finite_sum']} "
                      f"pts_n={eff_len} | ||img||={n_img:.2f} ||pts||={n_pts:.2f}")

    # attach predictions (contiguous)
    pred_df = pd.DataFrame({"key": keys, "pred_id": pred_ids})
    probs_np = np.array(prob_rows) if len(prob_rows) else np.zeros((0, num_classes))
    for k in range(num_classes):
        pred_df[f"prob_{k}"] = probs_np[:, k] if probs_np.shape[0] else []

    # map contiguous -> RAW (0..5 or pruned subset)
    pred_df["predicted_class"] = pred_df["pred_id"].map(inv_map).astype(int)
    out_gdf = crowns.merge(pred_df, on="key", how="left")

    # save predictions
    out_path = Path(OUT_GPKG)
    if out_path.exists():
        try: out_path.unlink()
        except Exception: pass
    out_gdf.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")
    print(f"Wrote {OUT_GPKG} (layer='{OUT_LAYER}') with {len(out_gdf)} crowns.")

    # diagnostics
    pd.DataFrame(diags).to_csv(DIAG_CSV, index=False)
    print(f"Diagnostics -> {DIAG_CSV}")

    # full debug dump
    debug_df = pd.DataFrame({
        "key": keys,
        "pred_id": pred_ids,
        "pred_prob": [max(p) for p in prob_rows],
        "entropy": entropies,
    })
    for c in range(num_classes):
        debug_df[f"prob_{c}"] = probs_np[:, c] if probs_np.shape[0] else []
    debug_df.to_csv(DEBUG_FULL_CSV, index=False)
    print(f"Full debug -> {DEBUG_FULL_CSV}")

    # quick summary
    if len(pred_ids):
        from collections import Counter
        hist_contig = dict(Counter(pred_ids))
        print(f"Pred class histogram (contiguous ids): {hist_contig}")
        hist_raw = dict(Counter([int(inv_map[c]) for c in pred_ids]))
        print(f"Pred class histogram (RAW labels):     {hist_raw}")
        if len(prob_rows):
            top2 = np.sort(np.array(prob_rows), axis=1)[:, -2:]
            print(f"Mean top-2 probs: top1={top2[:,1].mean():.3f}, top2={top2[:,0].mean():.3f}")
        n = max(len(pred_ids), 1)
        print("Mean logits (IMG only):", np.round((logits_sum_img / n).tolist(), 3))
        print("Mean logits (PTS only):", np.round((logits_sum_pts / n).tolist(), 3))
        print("Mean logits (blended): ", np.round((logits_sum / n).tolist(), 3))
        print(f"Feature norms — img: mean={np.mean(feat_img_norms):.2f}±{np.std(feat_img_norms):.2f} | "
              f"pts: mean={np.mean(feat_pts_norms):.2f}±{np.std(feat_pts_norms):.2f} | "
              f"mean pred prob={np.mean(sample_probs):.3f} | mean entropy={np.mean(entropies):.3f}")

    # evaluation
    if AOI_LABELS_GPKG and Path(AOI_LABELS_GPKG).exists():
        try:
            labels_gdf = gpd.read_file(AOI_LABELS_GPKG, layer=AOI_LAYER) if AOI_LAYER else gpd.read_file(AOI_LABELS_GPKG)
            if labels_gdf.crs != out_gdf.crs:
                labels_gdf = labels_gdf.to_crs(out_gdf.crs)
            pairs_df, cls_report, cm_df = _eval_match_and_report(out_gdf, labels_gdf, inv_map, iou_thr=EVAL_IOU_THR)
            pairs_df.to_csv(EVAL_MATCH_CSV, index=False)
            with open(EVAL_REPORT_TXT, "w") as f:
                f.write(cls_report + "\n")
            cm_df.to_csv(EVAL_CONFMAT_CSV)
            print("\n=== Classification report (RAW labels, matched by key or IoU) ===")
            print(cls_report)
            print("Confusion matrix (rows=ref, cols=pred):")
            print(cm_df)
            print(f"\nSaved matches -> {EVAL_MATCH_CSV}")
            print(f"Saved report  -> {EVAL_REPORT_TXT}")
            print(f"Saved matrix  -> {EVAL_CONFMAT_CSV}")
        except Exception as e:
            print(f"[WARN] Evaluation skipped: {e}")
    else:
        print("[INFO] AOI_LABELS_GPKG not set or missing; skipping evaluation.")

#
# import os, json
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import rasterio
# from rasterio.mask import mask as rio_mask
# from rasterio.windows import from_bounds
# from shapely.geometry import mapping
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
#
# from pyforestscan.handlers import read_lidar
# from tqdm import tqdm
#
# # ===== NEW: metrics =====
# try:
#     from sklearn.metrics import classification_report, confusion_matrix
#     _HAVE_SK = True
# except Exception:
#     _HAVE_SK = False
#
# # =======================
# # CONFIG — EDIT THESE
# # =======================
# PRED_CROWNS_GPKG  = "/Users/iosefa/repos/sam/cnn_predicted_ss1_overlap.gpkg"
# PRED_CROWNS_LAYER = "crowns_predicted__crowns_pred"
# BIG_WV3_TIF       = "/Users/iosefa/repos/sam/area51_subset1.tif"
#
# # EPT source (STRICT; CRS must match crowns/raster)
# EPT_JSON          = "/Users/iosefa/Downloads/ept6635/ept.json"
# EPT_SRS           = "EPSG:6635"
#
# # Classifier (4-class-pruned or 6-class)
# MODEL_PTH         = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft.pth"
# MODEL_META        = "/Users/iosefa/repos/sam/cnn_crowns_resnet34_mlp_pruned_ft.meta.json"
#
# # AOI labels (1-indexed 'class'); mapped to contiguous ids via meta["class_map"]
# AOI_LABELS_GPKG   = "/Users/iosefa/repos/sam/overlap_labels.gpkg"
# AOI_LAYER         = None
#
# # Outputs
# OUT_GPKG          = "/Users/iosefa/repos/sam/predicted_cnn_ss1.gpkg"
# OUT_LAYER         = "predicted_cnn"
# DIAG_CSV          = Path(OUT_GPKG).with_suffix(".diag.csv")
# DEBUG_FULL_CSV    = Path(OUT_GPKG).with_suffix(".debug_full.csv")
#
# # ===== NEW: evaluation outputs =====
# EVAL_IOU_THR      = 0.5  # IoU threshold if we need spatial matching
# EVAL_PREFIX       = Path(OUT_GPKG).with_suffix("")  # base path for eval outputs
# EVAL_REPORT_TXT   = Path(str(EVAL_PREFIX) + ".cls_report.txt")
# EVAL_CONFMAT_CSV  = Path(str(EVAL_PREFIX) + ".confusion_matrix.csv")
# EVAL_MATCH_CSV    = Path(str(EVAL_PREFIX) + ".matches.csv")
#
# # Image chip recovery/diagnostics
# MIN_FINITE_PIX    = 50
# BUFFER_METERS     = 0.5
# FALLBACK_SIZE_M   = 2.0
# REPORT_EVERY      = 25
#
# # ====== Reproduce training points behavior ======
# PAD_TO_MAX        = False   # False = downsample only; True = pad to max_points
#
# # ====== Calibration / priors / TTA / blending ======
# USE_PRIOR_CORRECTION = True
# PRIOR_MIX_ALPHA = 0.70       # ↑ stronger pull to AOI
# PRIOR_SCALE     = 1.00       # ↑ stronger effect (still capped)
# MAX_BIAS_ABS    = 1.20       # ↑ allow larger cap if AOI is very different
# TEMP            = 1.7
#
# # Test-time augmentation
# ENABLE_TTA            = True
# TTA_IMG_FLIPS         = [None, "h", "v", "hv"]
# TTA_PTS_ROT_K         = 8
#
# # Branch blending (base; adapted per-sample)
# W_IMG = 0.35
# W_PTS = 0.65
# IMG_SCALE   = 0.85
# PTS_SCALE   = 1.00
# LOGIT_CENTER = True
#
# # ================= Anti-collapse controls (tunable) =================
# CLASS3_ID = 3                 # contiguous id of the problematic class in pruned model
# CLASS3_MARGIN_MIN   = 0.18
# BRANCH_CONF_MIN     = 0.22
# TTA_CLASS3_VOTE_MIN = 0.55
# GLOBAL_SKEW_TRIGGER = 0.80
# GLOBAL_NUDGE_STEP   = 0.10
# GLOBAL_NUDGE_MAX    = 0.60
# GLOBAL_WARMUP       = 20
# STATIC_CLASS3_NUDGE = 0.00
#
# # =======================
# # RUNTIME DEVICE
# # =======================
# def get_device():
#     if torch.backends.mps.is_available(): return torch.device("mps")
#     if torch.cuda.is_available():         return torch.device("cuda")
#     return torch.device("cpu")
#
# # =======================
# # Helpers
# # =======================
# def entropy_np(p):
#     p = np.clip(p, 1e-9, 1.0)
#     return float(-(p * np.log(p)).sum())
#
# def margin_top2(p):
#     s = np.sort(p)
#     return float(s[-1] - s[-2])
#
# def adapt_weights(n_img, n_pts, base_img=0.30, base_pts=0.70):
#     # Responsive tilt; pivot ~6, slope ~2
#     s = 1.0 / (1.0 + np.exp(-((n_pts - 6.0) / 2.0)))
#     w_pts = base_pts * 0.6 + 0.4 * s
#     w_pts = float(np.clip(w_pts, 0.25, 0.85))
#     w_img = 1.0 - w_pts
#     return w_img, w_pts
#
# # =======================
# # MODEL
# # =======================
# class CombinedNet(nn.Module):
#     def __init__(self, num_classes, freeze_resnet=False, resnet="resnet34", dual_pool=True, p_drop=0.2):
#         super().__init__()
#         if resnet == "resnet18":
#             base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#             img_dim = 512
#         else:
#             base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
#             img_dim = 512
#         if freeze_resnet:
#             for p in base.parameters():
#                 p.requires_grad = False
#         self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
#
#         self.fc1 = nn.Linear(3, 64, bias=True)
#         self.fc2 = nn.Linear(64, 128, bias=True)
#         self.fc3 = nn.Linear(128, 256, bias=True)
#
#         self.dual_pool = dual_pool
#         pts_in = 512 if dual_pool else 256
#         self.fc4 = nn.Linear(pts_in, 128, bias=True)
#         self.fc5 = nn.Linear(128, 64, bias=True)
#
#         self.dropout = nn.Dropout(p_drop)
#         self.classifier = nn.Linear(img_dim + 64, num_classes)
#
#     def forward(self, img, pts, lengths=None, return_feats=False, branch_mask=None):
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
#             h_max = h.masked_fill(mask.unsqueeze(-1), float("-inf")).max(dim=1).values
#             if self.dual_pool:
#                 h_sum = h.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1)
#                 denom = lengths.clamp_min(1).unsqueeze(1).to(h.dtype)
#                 h_mean = h_sum / denom
#                 h_max[~torch.isfinite(h_max)] = 0.0
#                 h = torch.cat([h_max, h_mean], dim=1)        # (B,512)
#             else:
#                 h_max[~torch.isfinite(h_max)] = 0.0
#                 h = h_max                                    # (B,256)
#         else:
#             if self.dual_pool:
#                 h = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=1)
#             else:
#                 h = h.max(dim=1).values
#
#         h = F.relu(self.fc4(h))
#         h = self.dropout(h)
#         h = F.relu(self.fc5(h))                               # (B,64)
#
#         if branch_mask is not None:
#             if not branch_mask.get("img", True):
#                 x_img = torch.zeros_like(x_img)
#             if not branch_mask.get("pts", True):
#                 h = torch.zeros_like(h)
#
#         x = torch.cat([x_img, h], dim=1)                      # (B,576)
#         logits = self.classifier(x)
#         if return_feats:
#             return logits, x_img.norm(dim=1), h.norm(dim=1)
#         return logits
#
# # =======================
# # IMAGE HELPERS
# # =======================
# def robust_percentile_norm(band):
#     finite = np.isfinite(band)
#     if not finite.any():
#         return np.zeros_like(band, dtype=np.float32)
#     lo, hi = np.percentile(band[finite], [2, 98])
#     if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#         lo, hi = float(np.nanmin(band)), float(np.nanmax(band))
#         if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#             lo, hi = 0.0, 1.0
#     band = np.clip(band, lo, hi)
#     band = (band - lo) / max(hi - lo, 1e-6)
#     return band.astype(np.float32)
#
# def _read_by_polygon(src, poly):
#     if not poly.is_valid:
#         poly = poly.buffer(0)
#     out, _ = rio_mask(src, [mapping(poly)], crop=True, filled=False, pad=True)
#     return np.where(out.mask, np.nan, out.data).astype(np.float32)
#
# def _read_by_buffered_polygon(src, poly, buffer_m):
#     return _read_by_polygon(src, poly.buffer(buffer_m))
#
# def _read_by_centroid_window(src, poly, win_size_m):
#     cx, cy = poly.centroid.x, poly.centroid.y
#     half = win_size_m / 2.0
#     minx, miny, maxx, maxy = cx - half, cy - half, cx + half, cy + half
#     win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
#     arr = src.read(window=win, boundless=True, fill_value=np.nan).astype(np.float32)
#     if src.nodata is not None and np.isfinite(src.nodata):
#         arr[arr == src.nodata] = np.nan
#     return arr
#
# def _select_bands_from_meta(data_chw, meta):
#     C = data_chw.shape[0]
#     if C == 9 and meta.get("band_selection_9") is not None:
#         idx = list(meta["band_selection_9"])
#     elif C == 8 and meta.get("band_selection_8") is not None:
#         idx = list(meta["band_selection_8"])
#     else:
#         idx = list(range(min(3, C)))
#     if max(idx) >= C:
#         raise ValueError(f"Requested bands {idx} not present in raster with {C} bands.")
#     return data_chw[idx, :, :], idx
#
# def make_crown_tensor(src, poly, meta_img_size, meta):
#     stats = {"stage":"poly", "finite_sum":0, "finite_per_band":[0,0,0],
#              "std_per_band":[np.nan,np.nan,np.nan], "err":"", "bands_idx":[]}
#
#     data = _read_by_polygon(src, poly)
#     if not np.isfinite(data).any():
#         stats["stage"] = "buffer_poly"
#         data = _read_by_buffered_polygon(src, poly, BUFFER_METERS)
#     if not np.isfinite(data).any():
#         stats["stage"] = "centroid_win"
#         data = _read_by_centroid_window(src, poly, FALLBACK_SIZE_M)
#
#     arr, used_idx = _select_bands_from_meta(data, meta)
#     stats["bands_idx"] = used_idx
#
#     fin = [int(np.isfinite(arr[k]).sum()) for k in range(arr.shape[0])]
#     stats["finite_per_band"] = fin
#     stats["finite_sum"] = int(sum(fin))
#     stats["std_per_band"] = [float(np.nanstd(arr[k])) for k in range(arr.shape[0])]
#
#     if stats["finite_sum"] < MIN_FINITE_PIX and stats["stage"] != "centroid_win":
#         stats["stage"] += "->centroid_win"
#         data2 = _read_by_centroid_window(src, poly, FALLBACK_SIZE_M)
#         arr2, used_idx2 = _select_bands_from_meta(data2, meta)
#         fin2 = [int(np.isfinite(arr2[k]).sum()) for k in range(arr2.shape[0])]
#         if sum(fin2) > stats["finite_sum"]:
#             arr = arr2
#             stats["bands_idx"] = used_idx2
#             stats["finite_per_band"] = fin2
#             stats["finite_sum"] = int(sum(fin2))
#             stats["std_per_band"] = [float(np.nanstd(arr2[k])) for k in range(arr2.shape[0])]
#
#     norm = np.stack([robust_percentile_norm(arr[c]) for c in range(arr.shape[0])], axis=0)
#     t = torch.from_numpy(norm).unsqueeze(0)
#     t = F.interpolate(t, size=(meta_img_size, meta_img_size), mode="bilinear", align_corners=False).squeeze(0)
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#     std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
#     t = ((t - mean) / std).float()
#     return t, stats
#
# # =======================
# # POINTS HELPERS
# # =======================
# def _pdal_struct_to_xyz(pc_obj):
#     arr = pc_obj[0] if isinstance(pc_obj, (list, tuple)) else pc_obj
#     if arr is None:
#         raise RuntimeError("read_lidar returned None.")
#
#     # PDAL often returns a structured numpy array with named fields
#     if getattr(arr, "dtype", None) is not None and arr.dtype.names is not None:
#         names_lower = [n.lower() for n in arr.dtype.names]
#         if all(n in names_lower for n in ("x", "y", "z")):
#             X = arr[arr.dtype.names[names_lower.index("x")]]
#             Y = arr[arr.dtype.names[names_lower.index("y")]]
#             Z = arr[arr.dtype.names[names_lower.index("z")]]
#         else:
#             f0, f1, f2 = arr.dtype.names[:3]
#             X = arr[f0]; Y = arr[f1]; Z = arr[f2]
#         xyz = np.stack([np.asarray(X), np.asarray(Y), np.asarray(Z)], axis=1).astype(np.float32)
#     else:
#         a = np.asarray(arr)
#         if a.ndim != 2 or a.shape[1] < 3:
#             raise RuntimeError(f"Unexpected point array shape from read_lidar: {a.shape}")
#         xyz = a[:, :3].astype(np.float32)
#
#     return xyz
#
# def fetch_points_for_polygon(poly, ept_json, ept_srs):
#     minx, miny, maxx, maxy = poly.bounds
#     bounds = ([minx, maxx], [miny, maxy])
#     pc = read_lidar(ept_json, ept_srs, bounds, crop_poly=True, poly=poly.wkt)
#     pts = _pdal_struct_to_xyz(pc)
#     if pts.shape[0] == 0:
#         raise RuntimeError(f"EPT returned 0 points for bounds={bounds}")
#     return pts
#
# def normalize_points_xy_center(points):
#     xy = points[:, :2]; z = points[:, 2:3]
#     xy_c = xy - xy.mean(axis=0, keepdims=True)
#     std = xy_c.std(axis=0, keepdims=True); std[std == 0] = 1e-6
#     xy_n = xy_c / std
#     z_std = z.std(); z_std = z_std if np.isfinite(z_std) and z_std > 0 else 1e-6
#     z_n = (z - z.mean()) / z_std
#     out = np.hstack([xy_n, z_n]).astype(np.float32)
#     if not np.isfinite(out).all():
#         raise RuntimeError("Non-finite values in normalized point set.")
#     return out
#
# def downsample_if_needed(points, n_cap):
#     if points.shape[0] > n_cap:
#         idx = np.random.choice(points.shape[0], n_cap, replace=False)
#         return points[idx, :].astype(np.float32)
#     return points.astype(np.float32)
#
# def pad_to_max(points, n_max):
#     if points.shape[0] >= n_max:
#         idx = np.random.choice(points.shape[0], n_max, replace=False)
#         return points[idx, :].astype(np.float32), n_max
#     add = np.random.choice(points.shape[0], n_max - points.shape[0], replace=True)
#     out = np.vstack([points, points[add, :]]).astype(np.float32)
#     return out, n_max
#
# # ---- TTA helpers ----
# def apply_img_flip(t_b1):
#     outs = []
#     for f in TTA_IMG_FLIPS:
#         if f is None:
#             outs.append(t_b1)
#         elif f == "h":
#             outs.append(torch.flip(t_b1, dims=[3]))
#         elif f == "v":
#             outs.append(torch.flip(t_b1, dims=[2]))
#         elif f == "hv":
#             outs.append(torch.flip(torch.flip(t_b1, dims=[2]), dims=[3]))
#     return outs
#
# def rotate_points_z(pts, theta):
#     c, s = np.cos(theta), np.sin(theta)
#     R = np.array([[c, -s],[s, c]], dtype=np.float32)
#     out = pts.copy()
#     out[:, :2] = pts[:, :2] @ R.T
#     return out
#
# # =======================
# # AOI priors (1-indexed -> contiguous ids)
# # =======================
# def expected_priors_from_gpkg(gpkg_path, layer_name, class_map):
#     if not Path(gpkg_path).exists():
#         return None
#     g = gpd.read_file(gpkg_path, layer=layer_name) if layer_name else gpd.read_file(gpkg_path)
#     if "class" not in g.columns:
#         return None
#     labels_raw0 = pd.to_numeric(g["class"], errors="coerce").dropna().astype(int) - 1  # 1-index -> 0-index raw
#     raw2contig = {int(k): int(v) for k, v in class_map.items()} if isinstance(class_map, dict) else class_map
#     contig_ids = [raw2contig[r] for r in labels_raw0 if r in raw2contig]
#     if not contig_ids:
#         return None
#     num_classes = max(raw2contig.values()) + 1
#     counts = np.bincount(np.array(contig_ids, dtype=np.int64), minlength=num_classes).astype(np.float64)
#     priors = counts / np.clip(counts.sum(), 1e-12, None)
#     print(f"AOI prior counts (contiguous ids 0..{num_classes-1}): {counts.tolist()}")
#     print(f"AOI priors: {np.round(priors, 4).tolist()}")
#     return priors
#
# # ===== NEW: evaluation helpers =====
# def _ensure_int_series(s):
#     return pd.to_numeric(s, errors="coerce").dropna().astype(int)
#
# def _eval_match_and_report(pred_gdf, labels_gdf, inv_map, iou_thr=0.5):
#     """
#     Returns (pairs_df, text_report, conf_mat_df)
#     - pairs_df columns: key, pred_raw, ref_raw, iou
#     """
#     # predicted RAW labels already in 'predicted_class'
#     if "predicted_class" not in pred_gdf.columns:
#         raise RuntimeError("pred_gdf missing 'predicted_class'")
#
#     # convert AOI label to 0-index RAW
#     if "class" not in labels_gdf.columns:
#         raise RuntimeError("labels_gdf missing 'class'")
#     labels_gdf = labels_gdf.copy()
#     labels_gdf["ref_raw"] = _ensure_int_series(labels_gdf["class"]) - 1
#
#     # Fast path: join on 'key' if both have it
#     if "key" in pred_gdf.columns and "key" in labels_gdf.columns:
#         merged = pred_gdf[["key", "predicted_class", "geometry"]].merge(
#             labels_gdf[["key", "ref_raw", "geometry"]], on="key", how="inner", suffixes=("_pred","_ref")
#         )
#         # If geometries differ, compute IoU and filter
#         merged = gpd.GeoDataFrame(merged, geometry="geometry_pred", crs=pred_gdf.crs).rename(columns={"geometry_ref":"geom_ref"})
#         merged["geom_ref"] = merged["geom_ref"]
#         inter_area = merged["geometry_pred"].intersection(merged["geom_ref"]).area
#         union_area = merged["geometry_pred"].union(merged["geom_ref"]).area
#         iou = np.divide(inter_area, np.clip(union_area, 1e-12, None))
#         merged["iou"] = iou
#         pairs = merged[["key", "predicted_class", "ref_raw", "iou"]].rename(
#             columns={"predicted_class":"pred_raw"}
#         )
#         pairs = pairs[pairs["iou"] >= iou_thr].reset_index(drop=True)
#     else:
#         # Spatial match by max IoU (slower but robust)
#         pred = pred_gdf.copy()
#         lbl  = labels_gdf.copy()
#         if pred.crs != lbl.crs:
#             lbl = lbl.to_crs(pred.crs)
#         pred = pred.reset_index(drop=True).reset_index(names="pid")
#         lbl  = lbl.reset_index(drop=True).reset_index(names="vid")
#
#         # coarse sjoin to limit pairs
#         cand = gpd.sjoin(pred[["pid","geometry"]], lbl[["vid","geometry"]], how="inner", predicate="intersects").drop(columns=["index_right"])
#         if len(cand) == 0:
#             raise RuntimeError("No overlapping geometries found for evaluation.")
#         # compute IoU per candidate with overlays
#         pred_sub = pred.loc[cand["pid"], ["pid","geometry"]].rename(columns={"geometry":"geom_pred"})
#         lbl_sub  = lbl.loc[cand["vid"], ["vid","geometry"]].rename(columns={"geometry":"geom_ref"})
#         merged = cand.merge(pred_sub, on="pid").merge(lbl_sub, on="vid")
#         inter = merged.apply(lambda r: r["geom_pred"].intersection(r["geom_ref"]), axis=1)
#         inter_area = inter.area
#         union_area = (merged["geom_pred"].area + merged["geom_ref"].area - inter_area)
#         merged["iou"] = np.divide(inter_area, np.clip(union_area, 1e-12, None))
#
#         # for each pid, keep best vid by IoU
#         merged = merged.sort_values(["pid","iou"], ascending=[True, False]).groupby("pid", as_index=False).first()
#         keep = merged[merged["iou"] >= iou_thr]
#
#         pairs = keep.merge(pred[["pid","predicted_class"]], on="pid").merge(lbl[["vid","ref_raw"]], on="vid")
#         pairs = pairs.rename(columns={"predicted_class":"pred_raw"})[["pid","vid","iou","pred_raw","ref_raw"]]
#         pairs = pairs.rename(columns={"pid":"key"})  # 'key' stand-in
#
#     if len(pairs) == 0:
#         raise RuntimeError(f"No matches at IoU ≥ {iou_thr}. Cannot produce classification report.")
#
#     # sklearn report
#     if not _HAVE_SK:
#         raise RuntimeError("scikit-learn not installed. `pip install scikit-learn` to get classification reports.")
#
#     y_true = pairs["ref_raw"].to_numpy()
#     y_pred = pairs["pred_raw"].to_numpy()
#
#     labels_all = np.unique(np.concatenate([y_true, y_pred]))
#     report_str = classification_report(y_true, y_pred, labels=labels_all, digits=3)
#     cm = confusion_matrix(y_true, y_pred, labels=labels_all)
#     cm_df = pd.DataFrame(cm, index=[f"ref_{i}" for i in labels_all], columns=[f"pred_{i}" for i in labels_all])
#
#     return pairs, report_str, cm_df
#
# # =======================
# # MAIN
# # =======================
# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method("spawn", force=True)
#     device = get_device()
#     print(f"Device: {device}")
#
#     # Load meta
#     with open(MODEL_META, "r") as f:
#         meta = json.load(f)
#     num_classes = int(meta["num_classes"])
#     class_map   = {int(k): int(v) for k, v in meta["class_map"].items()} if isinstance(meta["class_map"], dict) else meta["class_map"]
#     inv_map     = {v:k for k,v in class_map.items()}  # contiguous -> RAW
#     img_size    = int(meta["img_size"])
#     max_points  = int(meta.get("max_points", 2048))
#
#     # Inspect checkpoint
#     state = torch.load(MODEL_PTH, map_location="cpu")
#     dual_pool = True
#     if "fc4.weight" in state:
#         dual_pool = (state["fc4.weight"].shape[1] == 512)
#     arch = str(meta.get("architecture", "")).lower()
#     resnet_kind = "resnet34" if "resnet34" in arch else ("resnet18" if "resnet18" in arch else "resnet34")
#
#     # Build and load
#     model = CombinedNet(num_classes=num_classes,
#                         freeze_resnet=meta.get("freeze_resnet", False),
#                         resnet=resnet_kind,
#                         dual_pool=dual_pool).to(device)
#     model.load_state_dict(state, strict=False)
#     model.eval()
#     print(f"Loaded model with dual_pool={dual_pool} (fc4 in_features = {'512' if dual_pool else '256'}), backbone={resnet_kind}")
#
#     # ---- Priors (tempered) ----
#     train_log_priors = np.array(meta.get("log_priors", [0.0]*num_classes), dtype=np.float32)
#     train_priors = np.exp(train_log_priors); train_priors /= np.clip(train_priors.sum(), 1e-12, None)
#
#     aoi_priors = expected_priors_from_gpkg(AOI_LABELS_GPKG, AOI_LAYER, class_map)
#     if USE_PRIOR_CORRECTION and aoi_priors is not None:
#         p_eff = (1.0 - PRIOR_MIX_ALPHA) * train_priors + PRIOR_MIX_ALPHA * aoi_priors
#         p_eff = p_eff / np.clip(p_eff.sum(), 1e-12, None)
#         raw_bias = np.log(np.clip(p_eff, 1e-6, 1.0)) - np.log(np.clip(train_priors, 1e-6, 1.0))
#         scaled_bias = PRIOR_SCALE * raw_bias
#         scaled_bias = np.clip(scaled_bias, -MAX_BIAS_ABS, MAX_BIAS_ABS)
#         prior_bias_corr = torch.tensor(-scaled_bias, dtype=torch.float32, device=device)  # subtract later
#         print("Using tempered prior correction:")
#         print(f"  train_priors   = {np.round(train_priors,4).tolist()}")
#         print(f"  aoi_priors     = {np.round(aoi_priors,4).tolist()}")
#         print(f"  mixed_priors   = {np.round(p_eff,4).tolist()}")
#         print(f"  raw_bias       = {np.round(raw_bias,3).tolist()}")
#         print(f"  scaled&clipped = {np.round(scaled_bias,3).tolist()} | TEMP={TEMP}")
#     else:
#         prior_bias_corr = torch.zeros(num_classes, dtype=torch.float32, device=device)
#         print(f"Prior correction disabled or AOI priors missing. TEMP={TEMP}")
#
#     # Optional static nudge vs class-3
#     static_corr = torch.zeros(num_classes, dtype=torch.float32, device=device)
#     if STATIC_CLASS3_NUDGE > 0 and CLASS3_ID < num_classes:
#         static_corr[CLASS3_ID] = float(STATIC_CLASS3_NUDGE)
#
#     # Global skew brake state
#     global_counts = np.zeros(num_classes, dtype=np.int64)
#     global_nudge  = np.zeros(num_classes, dtype=np.float32)  # in logit units
#
#     # Load predicted crowns
#     crowns = gpd.read_file(PRED_CROWNS_GPKG, layer=PRED_CROWNS_LAYER)
#     if crowns.crs is None:
#         raise RuntimeError("Predicted crowns GPKG has no CRS.")
#     print(f"Loaded {len(crowns)} predicted crowns.")
#
#     if not Path(EPT_JSON).exists():
#         raise RuntimeError(f"EPT JSON not found: {EPT_JSON}")
#     if str(crowns.crs) != EPT_SRS:
#         raise RuntimeError(f"CRS mismatch. crowns.crs='{crowns.crs}' but EPT_SRS='{EPT_SRS}'.")
#     print(f"EPT mode OK. ept_json='{EPT_JSON}', srs='{EPT_SRS}'")
#
#     if "key" not in crowns.columns:
#         crowns["key"] = [f"crown_{i}" for i in range(len(crowns))]
#
#     keys, pred_ids, prob_rows = [], [], []
#     diags = []
#     feat_img_norms, feat_pts_norms, sample_probs = [], [], []
#     logits_sum = np.zeros(num_classes, dtype=np.float64)
#     logits_sum_img = np.zeros(num_classes, dtype=np.float64)
#     logits_sum_pts = np.zeros(num_classes, dtype=np.float64)
#     entropies = []
#     first_print_done = False
#
#     with rasterio.open(BIG_WV3_TIF) as src:
#         if src.crs is None:
#             raise RuntimeError("Raster has no CRS.")
#         if str(src.crs) != EPT_SRS:
#             raise RuntimeError(f"Raster CRS '{src.crs}' must equal EPT_SRS '{EPT_SRS}'.")
#         if src.crs != crowns.crs:
#             raise RuntimeError("Raster CRS and crowns CRS differ (no reprojection).")
#
#         for i, row in tqdm(crowns.iterrows(), total=len(crowns), desc="Classifying crowns"):
#             key  = row["key"]
#             geom = row.geometry
#             if geom is None or geom.is_empty:
#                 raise RuntimeError(f"Empty geometry for key={key}")
#
#             # IMAGE
#             img_t, stats = make_crown_tensor(src, geom, img_size, meta)
#             if stats["finite_sum"] == 0:
#                 raise RuntimeError(f"No finite pixels for key={key}")
#
#             if not first_print_done:
#                 print(f"Using bands (0-based) from meta: {stats['bands_idx']}")
#                 first_print_done = True
#
#             # POINTS
#             pts = fetch_points_for_polygon(geom, EPT_JSON, EPT_SRS)
#             pts = normalize_points_xy_center(pts)
#             if PAD_TO_MAX:
#                 pts, eff_len = pad_to_max(pts, max_points)
#             else:
#                 pts = downsample_if_needed(pts, max_points)
#                 eff_len = pts.shape[0]
#
#             img_b1 = img_t.unsqueeze(0).to(device)
#             pts_np = pts.astype(np.float32)
#
#             # ---- helper: one forward producing blended & per-branch calibrated logits
#             def blend_logits(li, lp, w_img_dyn, w_pts_dyn):
#                 li = IMG_SCALE * li
#                 lp = PTS_SCALE * lp
#                 lb = w_img_dyn * li + w_pts_dyn * lp
#                 if LOGIT_CENTER:
#                     lb = lb - lb.mean(dim=1, keepdim=True)
#                 corr = prior_bias_corr + static_corr + torch.from_numpy(global_nudge).to(lb.device)
#                 lb = (lb - corr) / float(TEMP)
#                 li = (li - corr) / float(TEMP)
#                 lp = (lp - corr) / float(TEMP)
#                 return lb, li, lp
#
#             # base view (for dynamic weights + diagnostics)
#             length_t0 = torch.tensor([pts_np.shape[0]], dtype=torch.long, device=device)
#             pts_t0    = torch.from_numpy(pts_np).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 lg_img0 = model(img_b1, pts_t0, lengths=length_t0, branch_mask={"img": True,  "pts": False})
#                 lg_pts0 = model(img_b1, pts_t0, lengths=length_t0, branch_mask={"img": False, "pts": True})
#                 _, n_img_t0, n_pts_t0 = model(img_b1, pts_t0, lengths=length_t0, return_feats=True)
#             n_img = float(n_img_t0.item()); n_pts = float(n_pts_t0.item())
#             w_img_dyn, w_pts_dyn = adapt_weights(n_img, n_pts, base_img=W_IMG, base_pts=W_PTS)
#
#             # TTA views
#             img_views = apply_img_flip(img_b1) if ENABLE_TTA else [img_b1]
#             angs = [2*np.pi*k/float(TTA_PTS_ROT_K) for k in range(TTA_PTS_ROT_K)] if (ENABLE_TTA and TTA_PTS_ROT_K and TTA_PTS_ROT_K > 1) else [0.0]
#
#             logits_accum = []
#             tta_preds = []
#             with torch.no_grad():
#                 for im in img_views:
#                     for th in angs:
#                         try:
#                             pts_aug = rotate_points_z(pts_np, th).astype(np.float32)
#                             if pts_aug.shape[0] == 0 or not np.isfinite(pts_aug).all():
#                                 continue
#                             lt = torch.tensor([pts_aug.shape[0]], dtype=torch.long, device=device)
#                             pt = torch.from_numpy(pts_aug).unsqueeze(0).to(device)
#                             li = model(im, pt, lengths=lt, branch_mask={"img": True,  "pts": False})
#                             lp = model(im, pt, lengths=lt, branch_mask={"img": False, "pts": True})
#                             lb, _, _ = blend_logits(li, lp, w_img_dyn, w_pts_dyn)
#                             if lb is not None and torch.isfinite(lb).all():
#                                 logits_accum.append(lb)
#                                 tta_preds.append(int(lb.argmax(dim=1).item()))
#                         except Exception:
#                             continue
#
#             # fallback if TTA produced nothing
#             if len(logits_accum) == 0:
#                 lb_fallback, _, _ = blend_logits(lg_img0, lg_pts0, w_img_dyn, w_pts_dyn)
#                 logits_accum = [lb_fallback]
#                 tta_preds = [int(lb_fallback.argmax(dim=1).item())]
#
#             logits = torch.stack(logits_accum, dim=0).mean(dim=0)   # (1,C)
#             probs_blend = F.softmax(logits, dim=1).cpu().numpy()[0]
#             pred_blend  = int(probs_blend.argmax())
#             m_blend     = margin_top2(probs_blend)
#
#             # per-branch calibrated on base view (for guard/diagnostics)
#             lb0, li0, lp0 = blend_logits(lg_img0, lg_pts0, w_img_dyn, w_pts_dyn)
#             probs_img = F.softmax(li0, dim=1).cpu().numpy()[0]
#             probs_pts = F.softmax(lp0, dim=1).cpu().numpy()[0]
#             pred_img = int(probs_img.argmax()); m_img = margin_top2(probs_img)
#             pred_pts = int(probs_pts.argmax()); m_pts = margin_top2(probs_pts)
#
#             # --------- class-3 TTA vote gate + margin guard ----------
#             pred = pred_blend
#             if pred_blend == CLASS3_ID:
#                 vote_share_3 = (np.array(tta_preds) == CLASS3_ID).mean() if len(tta_preds) else 1.0
#                 veto_vote = vote_share_3 < TTA_CLASS3_VOTE_MIN
#                 veto_margin = m_blend < CLASS3_MARGIN_MIN
#
#                 if veto_vote or veto_margin:
#                     logits_np = logits.cpu().numpy()[0].copy()
#                     logits_np[CLASS3_ID] = -1e9
#                     cand1 = int(logits_np.argmax())
#
#                     cand2, cand2_margin = None, -1.0
#                     if pred_img != CLASS3_ID and m_img > BRANCH_CONF_MIN:
#                         cand2, cand2_margin = pred_img, m_img
#                     if pred_pts != CLASS3_ID and m_pts > BRANCH_CONF_MIN:
#                         if m_pts > cand2_margin:
#                             cand2, cand2_margin = pred_pts, m_pts
#
#                     if cand2 is not None:
#                         pred = cand2
#                     else:
#                         pred = cand1
#
#             # --------- global skew brake update ----------
#             global_counts[pred] += 1
#             seen = int(global_counts.sum())
#             if seen >= GLOBAL_WARMUP:
#                 frac = global_counts / np.clip(seen, 1, None)
#                 j = int(frac.argmax())
#                 if frac[j] >= GLOBAL_SKEW_TRIGGER and global_nudge[j] < GLOBAL_NUDGE_MAX:
#                     global_nudge[j] = float(min(GLOBAL_NUDGE_MAX, global_nudge[j] + GLOBAL_NUDGE_STEP))
#                     print(f"[Skew brake] After {seen} preds, class {j} at {frac[j]:.2f} -> increase global_nudge[{j}] to {global_nudge[j]:.2f}")
#
#             # diags
#             feat_img_norms.append(n_img)
#             feat_pts_norms.append(n_pts)
#             entropies.append(entropy_np(probs_blend))
#             logits_sum += logits.cpu().numpy()[0]
#             logits_sum_img += li0.detach().cpu().numpy()[0]
#             logits_sum_pts += lp0.detach().cpu().numpy()[0]
#
#             keys.append(key)
#             pred_ids.append(pred)
#             prob_rows.append(probs_blend.tolist())
#             sample_probs.append(float(probs_blend[pred]))
#
#             diags.append({
#                 "key": key,
#                 "stage": stats["stage"],
#                 "finite_sum": stats["finite_sum"],
#                 "finite_b0": stats["finite_per_band"][0],
#                 "finite_b1": stats["finite_per_band"][1],
#                 "finite_b2": stats["finite_per_band"][2],
#                 "std_b0": stats["std_per_band"][0],
#                 "std_b1": stats["std_per_band"][1],
#                 "std_b2": stats["std_per_band"][2],
#                 "err": "",
#                 "pts_mode": "ept_poly",
#                 "ept_bounds": str(([geom.bounds[0], geom.bounds[2]], [geom.bounds[1], geom.bounds[3]])),
#                 "pts_n_after_clip": int(eff_len),
#                 "feat_norm_img": n_img,
#                 "feat_norm_pts": n_pts,
#                 "pred_id": pred,
#                 "pred_prob": float(probs_blend[pred]),
#                 "entropy": entropies[-1],
#                 "w_img_dyn": w_img_dyn, "w_pts_dyn": w_pts_dyn,
#                 "m_blend": m_blend, "m_img": m_img, "m_pts": m_pts,
#                 "pred_blend": pred_blend, "pred_img": pred_img, "pred_pts": pred_pts,
#                 "tta_vote_share_class3": (np.array(tta_preds) == CLASS3_ID).mean() if len(tta_preds) else 1.0,
#                 "global_nudge": global_nudge.tolist(),
#             })
#
#             if (i+1) % REPORT_EVERY == 0:
#                 print(f"[{i+1}/{len(crowns)}] stage={stats['stage']} fin={stats['finite_sum']} "
#                       f"pts_n={eff_len} | ||img||={n_img:.2f} ||pts||={n_pts:.2f}")
#
#     # attach predictions (contiguous)
#     pred_df = pd.DataFrame({"key": keys, "pred_id": pred_ids})
#     probs_np = np.array(prob_rows) if len(prob_rows) else np.zeros((0, num_classes))
#     for k in range(num_classes):
#         pred_df[f"prob_{k}"] = probs_np[:, k] if probs_np.shape[0] else []
#
#     # map contiguous -> RAW (0..5 or pruned subset)
#     pred_df["predicted_class"] = pred_df["pred_id"].map(inv_map).astype(int)
#     out_gdf = crowns.merge(pred_df, on="key", how="left")
#
#     # save predictions
#     out_path = Path(OUT_GPKG)
#     if out_path.exists():
#         try: out_path.unlink()
#         except Exception: pass
#     out_gdf.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")
#     print(f"Wrote {OUT_GPKG} (layer='{OUT_LAYER}') with {len(out_gdf)} crowns.")
#
#     # diagnostics
#     pd.DataFrame(diags).to_csv(DIAG_CSV, index=False)
#     print(f"Diagnostics -> {DIAG_CSV}")
#
#     # full debug dump
#     debug_df = pd.DataFrame({
#         "key": keys,
#         "pred_id": pred_ids,
#         "pred_prob": [max(p) for p in prob_rows],
#         "entropy": entropies,
#     })
#     for c in range(num_classes):
#         debug_df[f"prob_{c}"] = probs_np[:, c] if probs_np.shape[0] else []
#     debug_df.to_csv(DEBUG_FULL_CSV, index=False)
#     print(f"Full debug -> {DEBUG_FULL_CSV}")
#
#     # quick summary (both contiguous and RAW)
#     if len(pred_ids):
#         from collections import Counter
#         hist_contig = dict(Counter(pred_ids))
#         print(f"Pred class histogram (contiguous ids): {hist_contig}")
#         hist_raw = dict(Counter([int(inv_map[c]) for c in pred_ids]))
#         print(f"Pred class histogram (RAW labels):     {hist_raw}")
#         if len(prob_rows):
#             top2 = np.sort(np.array(prob_rows), axis=1)[:, -2:]
#             print(f"Mean top-2 probs: top1={top2[:,1].mean():.3f}, top2={top2[:,0].mean():.3f}")
#         n = max(len(pred_ids), 1)
#         print("Mean logits (IMG only):", np.round((logits_sum_img / n).tolist(), 3))
#         print("Mean logits (PTS only):", np.round((logits_sum_pts / n).tolist(), 3))
#         print("Mean logits (blended): ", np.round((logits_sum / n).tolist(), 3))
#         print("Global nudge (final):  ", np.round(global_nudge.tolist(), 3))
#         print(f"Feature norms — img: mean={np.mean(feat_img_norms):.2f}±{np.std(feat_img_norms):.2f} | "
#               f"pts: mean={np.mean(feat_pts_norms):.2f}±{np.std(feat_pts_norms):.2f} | "
#               f"mean pred prob={np.mean(sample_probs):.3f} | mean entropy={np.mean(entropies):.3f}")
#
#     # =======================
#     # EVALUATION (classification report + confusion matrix)
#     # =======================
#     if AOI_LABELS_GPKG and Path(AOI_LABELS_GPKG).exists():
#         try:
#             labels_gdf = gpd.read_file(AOI_LABELS_GPKG, layer=AOI_LAYER) if AOI_LAYER else gpd.read_file(AOI_LABELS_GPKG)
#             # ensure CRS match
#             if labels_gdf.crs != out_gdf.crs:
#                 labels_gdf = labels_gdf.to_crs(out_gdf.crs)
#
#             pairs_df, cls_report, cm_df = _eval_match_and_report(out_gdf, labels_gdf, inv_map, iou_thr=EVAL_IOU_THR)
#
#             # Save matches + report + confusion matrix
#             pairs_df.to_csv(EVAL_MATCH_CSV, index=False)
#             with open(EVAL_REPORT_TXT, "w") as f:
#                 f.write(cls_report + "\n")
#             cm_df.to_csv(EVAL_CONFMAT_CSV)
#
#             # Print a concise summary
#             print("\n=== Classification report (RAW labels, matched by key or IoU) ===")
#             print(cls_report)
#             print("Confusion matrix (rows=ref, cols=pred):")
#             print(cm_df)
#             print(f"\nSaved matches -> {EVAL_MATCH_CSV}")
#             print(f"Saved report  -> {EVAL_REPORT_TXT}")
#             print(f"Saved matrix  -> {EVAL_CONFMAT_CSV}")
#         except Exception as e:
#             print(f"[WARN] Evaluation skipped: {e}")
#     else:
#         print("[INFO] AOI_LABELS_GPKG not set or missing; skipping evaluation.")
