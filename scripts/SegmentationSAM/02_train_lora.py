#!/usr/bin/env python3
"""
LoRA fine‑tuning of SAM’s mask‑decoder on 1024×1024 crown crops.
The only trainable parameters are LoRA adapters on q/v‑proj layers.
"""

from pathlib import Path
import argparse, math, warnings
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from peft import LoraConfig, get_peft_model

MEAN = torch.tensor([123.675,116.280,103.530]) / 255
STD  = torch.tensor([ 58.395, 57.120, 57.375]) / 255
RESZ = ResizeLongestSide(1024)          # identity for 1024² images

# ---------- dataset --------------------------------------------------
class Crowns(Dataset):
    def __init__(self, root: Path):
        self.imgs = sorted((root / "images").glob("*.png"))
        self.mroot = root / "masks"
        self.t = ToTensor()
    def __len__(self):  return len(self.imgs)
    def __getitem__(self, i):
        img = self.t(Image.open(self.imgs[i]).convert("RGB"))
        msk = self.t(Image.open(self.mroot /
                                f"{self.imgs[i].stem}_mask.png").convert("L"))
        return img, (msk > 0).float()

# ---------- SAM wrapper that duplicates the batch -------------------
class SAMWrap(nn.Module):
    """Freeze encoders, train mask‑decoder via LoRA.  No prompts."""
    def __init__(self, sam):
        super().__init__(); self.sam = sam
        self.config = {"use_return_dict": False,
                       "tie_word_embeddings": False}

    def forward(self, imgs, gt):
        """
        imgs : B×3×1024×1024  float32 in [0,1]
        gt   : B×1×1024×1024  float32 {0,1}
        """
        B = imgs.size(0)
        dev, dt = imgs.device, imgs.dtype
        n_mask = self.sam.mask_decoder.num_mask_tokens  # =4 (3 masks + IoU)

        # ── SAM normalisation ─────────────────────────────────────
        imgs = (imgs - MEAN.to(dev)[:, None, None]) / STD.to(dev)[:, None, None]
        imgs = RESZ.apply_image_torch(imgs)

        # ── encoder + position enc. ───────────────────────────────
        emb = self.sam.image_encoder(imgs)                 # B,C,64,64
        pe  = self.sam.prompt_encoder.get_dense_pe().to(dev, dt)

        # ── repeat per‑mask‑token along the **batch** axis ───────
        emb_rep = emb.repeat_interleave(n_mask, dim=0)     # B·n,C,64,64
        pe_rep  = pe.expand(emb_rep.shape[0], -1, -1, -1)  # B·n,C,64,64

        # empty sparse prompt (still needs correct batch dim)
        C = emb.shape[1]
        sparse = torch.zeros(emb_rep.shape[0], 0, C, device=dev, dtype=dt)

        # **dense prompt: all‑zeros with identical shape to `emb_rep`**
        dense  = torch.zeros_like(emb_rep)                 # B·n,C,64,64

        # ── mask decoder ─────────────────────────────────────────
        logits64, _, _ = self.sam.mask_decoder(
            image_embeddings         = emb_rep,
            image_pe                 = pe_rep,
            sparse_prompt_embeddings = sparse,
            dense_prompt_embeddings  = dense,   # exact same shape as `src`
            multimask_output         = False,
        )                                      # B·n,1,64,64

        # keep first mask token per original image
        logits64 = logits64.view(B, n_mask, 1, 64, 64)[:, 0]

        logits = F.interpolate(logits64, 1024, mode="bilinear",
                               align_corners=False)
        loss   = F.binary_cross_entropy_with_logits(logits, gt)
        return {"loss": loss}


# ---------- training loop -------------------------------------------
@torch.no_grad()
def run(model, loader, opt, dev, train=True):
    model.train() if train else model.eval()
    tot = n = 0
    for img, msk in loader:
        img, msk = img.to(dev), msk.to(dev)
        out = model(img, msk)
        if train:
            opt.zero_grad(); out["loss"].backward(); opt.step()
        tot += out["loss"].item() * img.size(0); n += img.size(0)
    return tot / n

# ---------- main -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data",       required=True)
    ap.add_argument("--out",        required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs",     type=int, default=4)
    ap.add_argument("--lr",     type=float, default=1e-4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    tr = DataLoader(Crowns(Path(args.data) / "train"), batch_size=args.bs,
                    shuffle=True)
    vl = DataLoader(Crowns(Path(args.data) / "val"),   batch_size=args.bs)

    base = sam_model_registry["vit_l"](checkpoint=args.checkpoint).to(dev)
    base.image_encoder.requires_grad_(False)
    base.prompt_encoder.requires_grad_(False)

    model = SAMWrap(base)
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                   target_modules=["q_proj", "v_proj"], bias="none")
    ).to(dev)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)

    best = math.inf
    out_p = Path(args.out).expanduser(); out_p.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr_l = run(model, tr, opt, dev, train=True)
        vl_l = run(model, vl, opt, dev, train=False)
        print(f"epoch {ep:02}/{args.epochs} | train {tr_l:.4f} | val {vl_l:.4f}")
        if vl_l < best:
            best = vl_l
            torch.save({"state": model.state_dict(),
                        "base":  args.checkpoint}, out_p)
            print("  ↳ saved best adapter")

    print("✓ finished – best val loss:", best)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()