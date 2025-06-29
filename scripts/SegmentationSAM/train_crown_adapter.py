#!/usr/bin/env python3
"""
train_crown_adapter.py  —  LoRA fine‑tuning of SAM’s mask‑decoder
on crown RGB + mask crops (1024×1024).
"""

from pathlib import Path
import argparse, math, warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from peft import LoraConfig, get_peft_model

# ---------- constants ------------------------------------------------
MEAN = torch.tensor([123.675, 116.280, 103.530]) / 255.0
STD  = torch.tensor([ 58.395,  57.120,  57.375]) / 255.0
RESIZE = ResizeLongestSide(1024)              # identity for 1024²

# ---------- SAM wrapper ----------------------------------------------
class SamWithConfig(nn.Module):
    """Wrap SAM, expose .config, train only the mask‑decoder."""
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.config = {"use_return_dict": False,
                       "tie_word_embeddings": False}

    def forward(self, imgs, gt_masks):
        """
        imgs     : B×3×1024×1024  float in [0,1]
        gt_masks : B×1×1024×1024  float {0,1}
        """
        B = imgs.shape[0]
        dev, dt = imgs.device, imgs.dtype

        # ── SAM image preprocessing ─────────────────────────────────
        imgs_n = (imgs - MEAN.to(dev)[:, None, None]) / STD.to(dev)[:, None, None]
        imgs_n = RESIZE.apply_image_torch(imgs_n)          # still 1024²

        img_emb = self.sam.image_encoder(imgs_n)           # B,C,64,64
        img_pe  = self.sam.prompt_encoder.get_dense_pe().to(dev, dt)  # 1,C,64,64

        C = img_emb.shape[1]

        # empty sparse prompt (B×0×C)
        sparse = torch.zeros(B, 0, C, device=dev, dtype=dt)

        # ◆ scalar dense prompt — B,C,1,1   (let decoder broadcast it)
        dense  = torch.zeros(B, C, 1, 1, device=dev, dtype=dt)

        logits_64, _, _ = self.sam.mask_decoder(
            image_embeddings         = img_emb,         # B,C,64,64
            image_pe                 = img_pe,          # 1,C,64,64  (broadcast)
            sparse_prompt_embeddings = sparse,
            dense_prompt_embeddings  = dense,
            multimask_output         = False,
        )                                                # B,1,64,64 logits

        logits_1024 = F.interpolate(logits_64, 1024, mode='bilinear',
                                    align_corners=False)
        loss = F.binary_cross_entropy_with_logits(logits_1024, gt_masks)
        return {"loss": loss, "logits": logits_1024}

    # forward unknown attributes to underlying SAM
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.sam, name)

# ---------- dataset --------------------------------------------------
class CrownDataset(Dataset):
    def __init__(self, split):
        self.imgs = sorted((split / 'images').glob('*.png'))
        self.mroot = split / 'masks'
        self.t = ToTensor()

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = self.t(Image.open(self.imgs[i]).convert('RGB'))
        msk = self.t(Image.open(self.mroot /
                                f"{self.imgs[i].stem}_mask.png").convert('L'))
        return img, (msk > 0).float()

@torch.no_grad()
def run_epoch(model, loader, optim, dev, train=True):
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    for img, msk in loader:
        img, msk = img.to(dev), msk.to(dev)
        out = model(img, msk)
        if train:
            optim.zero_grad(); out['loss'].backward(); optim.step()
        tot += out['loss'].item() * img.size(0); n += img.size(0)
    return tot / n

# ---------- main -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data-dir',   required=True)
    ap.add_argument('--out',        required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr',     type=float, default=1e-4)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    tr = CrownDataset(Path(args.data_dir)/'train')
    vl = CrownDataset(Path(args.data_dir)/'val')
    tr_ld = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    vl_ld = DataLoader(vl, batch_size=args.batch_size, shuffle=False)

    sam_base = sam_model_registry['vit_l'](checkpoint=args.checkpoint)
    sam_base.image_encoder.requires_grad_(False)
    sam_base.prompt_encoder.requires_grad_(False)
    sam_base.to(dev).eval()

    sam = SamWithConfig(sam_base)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'], bias='none')
    sam = get_peft_model(sam, lora_cfg).to(dev)

    opt = torch.optim.AdamW(
        (p for p in sam.parameters() if p.requires_grad), lr=args.lr)

    best = math.inf
    out_p = Path(args.out).expanduser()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs+1):
        tr_l = run_epoch(sam, tr_ld, opt, dev, train=True)
        vl_l = run_epoch(sam, vl_ld, opt, dev, train=False)
        print(f"epoch {ep:02}/{args.epochs} | train {tr_l:.4f} | val {vl_l:.4f}")
        if vl_l < best:
            best = vl_l
            sam.save_pretrained(out_p.with_suffix(''))
            torch.save({'lora_state_dict': sam.state_dict(),
                        'base_checkpoint': args.checkpoint}, out_p)
            print('  ↳ saved best adapter →', out_p)

    print('✓ finished – best val loss:', best)

# --------------------------------------------------------------------
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()