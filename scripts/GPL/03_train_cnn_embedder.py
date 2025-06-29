#!/usr/bin/env python3
"""
03_train_cnn_embedder.py  –  self-supervised encoder for crown chips
--------------------------------------------------------------------
* Loads 8-band WV-3 GeoTIFF chips with rasterio (no CHM band).
* SimCLR contrastive learning → 256-D embedding per crown.
* Saves .npy (matrix) and .csv (row-per-crown) side-by-side.
"""

from pathlib import Path
import random, json
from typing import Tuple, List

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, models

# ────────────── USER SETTINGS ──────────────────────────────────────
CHIPS_DIR   = Path("/Users/iosefa/repos/obia/docs/example_data/site_1/individual_crowns")

USE_RGB     = False                # True → 3-band, False → 8-band
RGB_BANDS   = (4, 2, 1)            # WV-3 order 1-indexed → R,G,B = 5,3,2

EMBED_DIM   = 256
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 2e-4
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_NPY = CHIPS_DIR.parent / "crown_embeddings.npy"
OUT_CSV = CHIPS_DIR.parent / "crown_embeddings.csv"
# ───────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────── DATASET ───────────────────────────────
class CrownChipDS(Dataset):
    """Load 8-band GeoTIFF chips, return two augmented views + id."""
    def __init__(self, root: Path, resize: int, rgb_only: bool):
        self.files  = sorted(root.glob("*.tif"))
        self.resize = resize
        self.rgb    = rgb_only

        self.pre_tf = T.Compose([
            T.Resize((resize, resize), antialias=True),
        ])

        self.aug_tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1) if self.rgb else nn.Identity(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def _load_chip(self, fp: Path) -> torch.Tensor:
        with rasterio.open(fp) as src:
            arr = src.read()                   # (bands, H, W)
        arr = arr.astype(np.float32)

        # Select bands
        if self.rgb:
            arr = arr[np.array(RGB_BANDS) - 1]          # 3 bands
        else:
            arr = arr[:8]                               # first 8 bands

        arr /= np.iinfo(src.dtypes[0]).max              # 0-1 normalisation
        img = torch.from_numpy(arr)                     # B × H × W
        img = self.pre_tf(img)                          # resize
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fp  = self.files[idx]
        img = self._load_chip(fp)
        v1  = self.aug_tf(img)
        v2  = self.aug_tf(img)
        return v1, v2, fp.stem


# ──────────────────────── ENCODER MODEL ────────────────────────────
def make_encoder(in_channels: int, embed_dim: int) -> nn.Module:
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if in_channels != 3:
        old_w = resnet.conv1.weight
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        with torch.no_grad():
            # copy RGB weights; random-init remaining channels
            resnet.conv1.weight[:, :3] = old_w
            if in_channels > 3:
                nn.init.kaiming_normal_(resnet.conv1.weight[:, 3:])
    feat_dim = resnet.fc.in_features
    backbone = nn.Sequential(*list(resnet.children())[:-1])   # drop fc
    projector = nn.Sequential(
        nn.Linear(feat_dim, embed_dim),
        nn.ReLU(inplace=True),
        nn.Linear(embed_dim, embed_dim),
    )

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.proj     = projector
        def forward(self, x):
            x = self.backbone(x)      # N × feat_dim × 1 × 1
            x = x.flatten(1)
            x = nn.functional.normalize(self.proj(x), dim=1)
            return x
    return Encoder()


# ─────────────────────── Contrastive loss ──────────────────────────
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.1):
    N  = z1.size(0)
    z  = torch.cat([z1, z2], dim=0)          # 2N × D
    sim = torch.mm(z, z.t()) / temp          # cosine sim
    mask = torch.eye(2*N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)            # no self-sim
    pos = torch.cat([torch.diag(sim,  N), torch.diag(sim, -N)])
    loss = -pos + torch.logsumexp(sim, dim=1)
    return loss.mean()


# ─────────────────────────── MAIN ──────────────────────────────────
def main():
    ds = CrownChipDS(CHIPS_DIR, IMG_SIZE, USE_RGB)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)

    in_ch  = 3 if USE_RGB else 8
    model  = make_encoder(in_ch, EMBED_DIM).to(DEVICE)
    opt    = optim.Adam(model.parameters(), lr=LR)

    print(f"• training encoder on {len(ds):,} chips  ({EPOCHS} epochs)")

    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for v1, v2, _ in loader:
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            z1, z2 = model(v1), model(v2)
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * v1.size(0)
        print(f"  epoch {ep+1:02d}/{EPOCHS}  loss={tot/len(ds):.4f}")

    # ── embeddings (deterministic pass) ────────────────────────────
    model.eval()
    embeds, ids = [], []
    with torch.no_grad():
        for v1, _, names in DataLoader(ds, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=0):
            emb = model(v1.to(DEVICE)).cpu().numpy()
            embeds.append(emb); ids.extend(names)
    X = np.vstack(embeds)
    np.save(OUT_NPY, X)
    pd.DataFrame(X, index=ids).to_csv(OUT_CSV)
    print("✓ embeddings:", X.shape, "→", OUT_NPY)
    print("✓ CSV saved →", OUT_CSV)


if __name__ == "__main__":
    main()