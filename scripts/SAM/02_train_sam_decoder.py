#!/usr/bin/env python3
"""
Fine‑tune the SAM‑ViT‑L mask‑decoder on the chips produced by step‑01.
"""
import json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

ROOT   = Path(__file__).parent
CHIPS  = ROOT / "data" / "chips"
CKPT   = "sam_vit_l_0b3195.pth"          # put file next to the script
OUT    = ROOT / "sam_crowns_decoder.pth"

IMG, ENC = 512, 1024
EPOCHS, LR = 15, 5e-5
DEV = "mps" if torch.backends.mps.is_available() \
      else "cuda" if torch.cuda.is_available() else "cpu"

class Chips(Dataset):
    def __init__(self):
        self.ids = sorted((CHIPS/"img").glob("*.npy"))
        self.rs  = ResizeLongestSide(ENC)
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        n = self.ids[i].stem
        img = np.load(CHIPS/"img"/f"{n}.npy")
        msk = np.load(CHIPS/"msk"/f"{n}.npy")
        pt  = np.array(json.load(open(CHIPS/"json"/f"{n}.json"))["point"])
        img = self.rs.apply_image(img.transpose(1,2,0)).transpose(2,0,1)/255.
        pt  = self.rs.apply_coords(pt[None], (IMG,IMG))[0]
        return (torch.tensor(img).float(),
                torch.tensor(msk[None]).float(),
                torch.tensor(pt).float())

dl = DataLoader(Chips(), batch_size=1, shuffle=True)
print("chips:", len(dl))

sam = sam_model_registry["vit_l"](checkpoint=CKPT).to(DEV)
sam.image_encoder.requires_grad_(False)
sam.prompt_encoder.requires_grad_(False)

opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=LR)
base_pe = sam.prompt_encoder.get_dense_pe().to(DEV)

def loss_fn(p, g):
    s = torch.sigmoid(p)
    dice = 1-(2*(s*g).sum((2,3))+1)/(s.sum((2,3))+g.sum((2,3))+1)
    return (dice + F.binary_cross_entropy_with_logits(p,g)).mean()

for ep in range(1, EPOCHS+1):
    tot = 0
    for img, msk, pt in dl:
        img, msk, pt = img.to(DEV), msk.to(DEV), pt.to(DEV)

        with torch.no_grad():
            emb = sam.image_encoder(sam.preprocess(img))

        sparse,_ = sam.prompt_encoder(
            points=(pt[:,None,:], torch.ones((1,1),device=DEV)),
            boxes=None, masks=None)

        logit_low,_ = sam.mask_decoder(
            image_embeddings=emb,
            image_pe=base_pe,               # broadcast automatically
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=base_pe,
            multimask_output=False)
        logit = torch.nn.functional.interpolate(
            logit_low, size=(IMG,IMG), mode="bilinear", align_corners=False)

        loss = loss_fn(logit, msk)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"epoch {ep:02d}/{EPOCHS}  loss={tot/len(dl):.4f}")

torch.save(sam.state_dict(), OUT)
print("✓ weights →", OUT)