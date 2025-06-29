#!/usr/bin/env python3
# 02_train_sam_finetune.py  – mini-batch 1, original mask tokens intact
# Works on mps / cuda / cpu

import json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# ───────── config ─────────────────────────────────────────────────
ROOT = Path("data/chips")
CKPT = "sam_vit_l_0b3195.pth"
OUT  = "sam_crowns_vit_l_decoder.pth"
DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
LR, EPOCHS = 5e-5, 15
IMG, ENC   = 512, 1024

# ───────── dataset ────────────────────────────────────────────────
class Chips(Dataset):
    def __init__(self):
        self.ids = sorted((ROOT / "img").glob("*.npy"))
        self.rs  = ResizeLongestSide(ENC)
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        stem = self.ids[i].stem
        img  = np.load(ROOT/"img"/f"{stem}.npy")
        msk  = np.load(ROOT/"msk"/f"{stem}.npy")
        pt   = np.array(json.load(open(ROOT/"json"/f"{stem}.json"))["point"])
        img  = self.rs.apply_image(img.transpose(1,2,0)).transpose(2,0,1)/255.
        pt   = self.rs.apply_coords(pt[None], (IMG,IMG))[0]
        return (torch.tensor(img).float(),
                torch.tensor(msk[None]).float(),
                torch.tensor(pt).float())

loader = DataLoader(Chips(), batch_size=1, shuffle=True, num_workers=0)
print("chips =", len(loader))

# ───────── model (original tokens) ────────────────────────────────
sam = sam_model_registry["vit_l"](checkpoint=CKPT).to(DEVICE)
sam.image_encoder.requires_grad_(False)
sam.prompt_encoder.requires_grad_(False)

opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=LR)
base_pe = sam.prompt_encoder.get_dense_pe().to(DEVICE)      # (1,C,H,W)

def loss_fn(pred, gt):
    p = torch.sigmoid(pred)
    dice = 1 - (2*(p*gt).sum((2,3))+1)/(p.sum((2,3))+gt.sum((2,3))+1)
    return (dice + F.binary_cross_entropy_with_logits(pred, gt)).mean()

# ───────── training loop ──────────────────────────────────────────
for ep in range(1, EPOCHS+1):
    sam.train(); tot = 0
    for img, msk, pt in loader:
        img, msk, pt = img.to(DEVICE), msk.to(DEVICE), pt.to(DEVICE)

        # frozen image encoder
        with torch.no_grad():
            emb = sam.image_encoder(sam.preprocess(img))

        # prompt tensors
        points = pt[:, None, :]                      # (B=1,1,2)
        labels = torch.ones((1,1), dtype=torch.int, device=DEVICE)
        sparse,_ = sam.prompt_encoder(points=(points, labels),
                                      boxes=None, masks=None)

        # figure out internal duplication factor
        dup = emb.size(0) // img.size(0)             # e.g. 8
        if dup > 1:
            sparse = sparse.repeat_interleave(dup, 0)          # (B*dup,…)
            dense  = base_pe.repeat(dup,1,1,1)                 # (B*dup,C,H,W)
            msk    = msk.repeat_interleave(dup, 0)             # (B*dup,1,H,W)
            emb    = emb                                       # already B*dup
            pe     = dense
        else:
            dense  = base_pe
            pe     = base_pe

        logit_low,_ = sam.mask_decoder(
            image_embeddings=emb,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logit = torch.nn.functional.interpolate(
            logit_low, size=(IMG,IMG), mode="bilinear", align_corners=False)

        loss = loss_fn(logit, msk)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()

    print(f"epoch {ep:02d}/{EPOCHS}  loss={tot/len(loader):.4f}")

torch.save(sam.state_dict(), OUT)
print("✓ weights saved →", OUT)