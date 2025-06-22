"""test_imagecas.py — volume‑level evaluation for ImageCas

* **Runs the whole test set** (default), `random`, or a specific volume prefix.
* Computes **mean per‑slice Dice** for each volume and prints a grand mean.
* **Only two volumes are saved**: the best‑Dice volume and the worst‑Dice volume.
  They each get:
    • `<pred_dir>/<vol>_pred.nii.gz`  – 3‑D mask
    • `<pred_dir>/<vol>_img.nii.gz`   – stacked CT image
    • `<pred_dir>/<vol>_overlay.png`  – 3‑slice overlay PNG

Usage
-----
```bash
python test_imagecas.py           # evaluate full test set, save best & worst
python test_imagecas.py random    # pick one random volume (still saved)
python test_imagecas.py extracted_801-1000_933.img.nii  # single volume
```
"""
from __future__ import annotations
import sys, random, logging, numpy as np, shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ───────── paths & constants ───────────────────────────────────────────────
ROOT_PATH = Path("/home/ubuntu/hist/TransUNet/data")
LIST_DIR  = Path("/home/ubuntu/hist/TransUNet/lists/lists_ImageCas")
CKPT_DIR  = Path("/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k")
PRED_DIR  = CKPT_DIR / "preds"
IMG_SIZE  = 224
NUM_CLASSES = 2
SEED = 1235
ARG = (sys.argv[1].lower() if len(sys.argv) > 1 else "all")

random.seed(SEED); np.random.seed(SEED)
TorchDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark, cudnn.deterministic = True, False

# ───────── model init ─────────────────────────────────────────────────────-
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg, SegmentationHead
cfg = CONFIGS_ViT_seg["R50-ViT-B_16"]
cfg.n_classes = NUM_CLASSES; cfg.n_skip = 3
cfg.patches.size = (16,16); cfg.patches.grid = (IMG_SIZE//16, IMG_SIZE//16)
net = ViT_seg(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES).to(TorchDevice)
net.segmentation_head = SegmentationHead(cfg["decoder_channels"][-1], NUM_CLASSES, 3).to(TorchDevice)
ckpt = CKPT_DIR / "best.pth"
if not ckpt.exists():
    raise FileNotFoundError(f"{ckpt} missing; train first.")
net.load_state_dict(torch.load(ckpt, map_location="cpu")); net.eval()
print(f"Loaded {ckpt.name}")

# ───────── dataset organisation ───────────────────────────────────────────-
from datasets.dataset_imagecas import ImageCas_dataset
ds = ImageCas_dataset(str(ROOT_PATH), str(LIST_DIR), split="test_vol", transform=None)

if ARG == "random":
    s = random.choice(ds.sample_list); VOL_FILTER = s.split("_slice")[0]
    print(f"🎲 Random volume: {VOL_FILTER}")
elif ARG in {"", "all"}:
    VOL_FILTER = ""
else:
    VOL_FILTER = ARG

vol_slices: dict[str, List[str]] = defaultdict(list)
for s in ds.sample_list:
    if VOL_FILTER and not s.startswith(VOL_FILTER):
        continue
    vol_slices[s.split("_slice")[0]].append(s)
if not vol_slices:
    raise ValueError(f"No volume matches '{VOL_FILTER}'.")
print(f"Evaluating {len(vol_slices)} volume(s)")

PRED_DIR.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(str(CKPT_DIR/"tb"/"infer_vol"))

# ───────── helpers ─────────────────────────────────────────────────────────

def model_predict(img_np: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(TorchDevice)
    with torch.no_grad():
        logits = net(t)
    return torch.argmax(torch.softmax(logits,1),1).cpu().numpy()[0]

def overlay(img, gt, pred, alpha=0.5):
    base = (img-img.min())/(img.ptp()+1e-5)
    rgb = np.stack([base]*3,-1)
    rgb[...,1] = np.where(gt>0, alpha*1+(1-alpha)*rgb[...,1], rgb[...,1])
    rgb[...,0] = np.where(pred>0,alpha*1+(1-alpha)*rgb[...,0],rgb[...,0])
    return (rgb*255).astype(np.uint8)

# ───────── evaluation loop ─────────────────────────────────────────────---
best_vol, worst_vol = None, None
best_dice, worst_dice = -1, 2
all_dices: List[float] = []
all_hd95: List[float] = []

for v_idx, (vid, slices) in enumerate(vol_slices.items()):
    slices.sort(key=lambda x:int(x.split("_slice")[-1].split(".npz")[0]))
    imgs, gts, preds = [], [], []
    for s in tqdm(slices, desc=vid):
        arr = np.load(ROOT_PATH/"val_processed_224"/s)
        img_np, gt_np = arr["image"], arr["label"]
        pred_np = model_predict(img_np)
        imgs.append(img_np); gts.append(gt_np); preds.append(pred_np)

    imgs, gts, preds = map(np.stack, (imgs,gts,preds))
    # per‑slice dice then mean
        # Dice per slice ---------------------------------------------------------
    slice_dice = [
        2*np.logical_and(preds[i]>0, gts[i]>0).sum() /
        ((preds[i]>0).sum() + (gts[i]>0).sum() + 1e-5)
        for i in range(len(slices))
    ]
    dice_vol = float(np.mean(slice_dice))

    # HD95 per slice -------------------------------------------------------
    from scipy.ndimage import distance_transform_edt as edt
    def _hd95(a,b):
        if a.sum()==0 and b.sum()==0: return 0.0
        if a.sum()==0 or b.sum()==0:  return 95.0
        dt_a = edt(1-a); dt_b = edt(1-b)
        return float(np.percentile(np.hstack([dt_a[b>0], dt_b[a>0]]), 95))
    slice_hd  = [_hd95(preds[i]>0, gts[i]>0) for i in range(len(slices))]
    hd95_vol  = float(np.mean(slice_hd))

    # aggregate metrics ----------------------------------------------------
    all_dices.append(dice_vol)
    all_hd95.append(hd95_vol)
    writer.add_scalar("dice_vol", dice_vol, v_idx)
    writer.add_scalar("hd95_vol", hd95_vol, v_idx)
    print(f"{vid}: Dice {dice_vol:.4f}  HD95 {hd95_vol:.2f}")

    # track best & worst ---------------------------------------------------
    def save_volume(vol_id, imgs, preds, gts):
        aff = np.eye(4)
        nib.save(nib.Nifti1Image(preds.astype(np.uint8), aff), PRED_DIR/f"{vol_id}_pred.nii.gz")
        nib.save(nib.Nifti1Image(imgs.astype(np.float32), aff),  PRED_DIR/f"{vol_id}_img.nii.gz")
        fg = np.where(gts.reshape(len(gts), -1).sum(1)>0)[0]
        sel = [fg[0], fg[len(fg)//2], fg[-1]] if len(fg)>=3 else list(fg)
        fig, ax = plt.subplots(1,len(sel), figsize=(4*len(sel),4))
        if len(sel)==1: ax=[ax]
        for a,idx in zip(ax,sel):
            a.imshow(overlay(imgs[idx], gts[idx], preds[idx])); a.axis('off'); a.set_title(idx)
        plt.tight_layout(); plt.savefig(PRED_DIR/f"{vol_id}_overlay.png",dpi=150); plt.close(fig)

    if dice_vol > best_dice:
        # remove previous best
        if best_vol:
            for f in PRED_DIR.glob(f"{best_vol}_*.*"):
                f.unlink()
        best_vol, best_dice = vid, dice_vol
        save_volume(vid, imgs, preds, gts)
    if dice_vol < worst_dice:
        if worst_vol and worst_vol!=best_vol:  # avoid deleting best
            for f in PRED_DIR.glob(f"{worst_vol}_*.*"):
                f.unlink()
        worst_vol, worst_dice = vid, dice_vol
        save_volume(vid, imgs, preds, gts)

# ───────── summary ─────────────────────────────────────────────────────---
mean_dice = np.mean(all_dices)
mean_hd95 = np.mean(all_hd95)
print("==============================")
print(f"Grand‑mean Dice over {len(all_dices)} volumes: {mean_dice:.4f}")
print(f"Grand‑mean HD95 over {len(all_hd95)} volumes: {mean_hd95:.2f}")
print(f"Best:  {best_vol}  Dice {best_dice:.4f}")
print(f"Worst: {worst_vol} Dice {worst_dice:.4f}")
print("Saved results for best & worst volumes to", PRED_DIR)

