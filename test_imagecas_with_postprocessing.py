"""
Enhanced test_imagecas.py with post-processing comparison
Replace your existing test_imagecas.py with this version
"""

from __future__ import annotations
import sys, random, logging, numpy as np, shutil
from pathlib import Path
from collections import defaultdict
from typing import List
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Add post-processing import
from postprocessing.coronary_postprocessing import (
    CoronaryPostProcessor, 
    calculate_dice_score, 
    calculate_hd95,
    create_comparison_overlay
)

# ───────── Configuration ───────────────────────────────────────────────────
ROOT_PATH = Path("/home/ubuntu/hist/TransUNet/data")
LIST_DIR  = Path("/home/ubuntu/hist/TransUNet/lists/lists_ImageCas")
CKPT_DIR  = Path("/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k")
PRED_DIR  = CKPT_DIR / "preds_with_postprocessing"
IMG_SIZE  = 224
NUM_CLASSES = 2
SEED = 1236
ARG = (sys.argv[1].lower() if len(sys.argv) > 1 else "all")

random.seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark, cudnn.deterministic = True, False

# ───────── Load Model ─────────────────────────────────────────────────────
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg, SegmentationHead
from utils import DiceLoss

cfg = CONFIGS_ViT_seg["R50-ViT-B_16"]
cfg.n_classes = NUM_CLASSES; cfg.n_skip = 3
cfg.patches.size = (16,16); cfg.patches.grid = (IMG_SIZE//16, IMG_SIZE//16)
net = ViT_seg(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
net.segmentation_head = SegmentationHead(cfg["decoder_channels"][-1], NUM_CLASSES, 3).to(DEVICE)

ckpt = CKPT_DIR / "best.pth"
if not ckpt.exists():
    raise FileNotFoundError(f"{ckpt} missing; train first.")
net.load_state_dict(torch.load(ckpt, map_location="cpu")); net.eval()
print(f"Loaded {ckpt.name}")

dice_fn = DiceLoss(n_classes=NUM_CLASSES).to(DEVICE)

# ───────── Initialize Post-Processor ─────────────────────────────────────
post_processor = CoronaryPostProcessor(
    min_component_size=50,
    closing_radius=2,
    opening_radius=1,
    max_gap_distance=3
)

# ───────── Load Dataset ───────────────────────────────────────────────────
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

print(f"Evaluating {len(vol_slices)} volume(s) with post-processing comparison")

PRED_DIR.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(str(CKPT_DIR/"tb"/"infer_vol_postproc"))

# ───────── Metrics Storage ─────────────────────────────────────────────────
methods = ["original", "basic", "advanced"]
all_metrics = {method: {"dice": [], "hd95": []} for method in methods}
best_vol, worst_vol, best_dice, worst_dice = None, None, -1.0, 2.0

# ───────── Evaluation Loop ─────────────────────────────────────────────────
for v_idx, (vid, slices) in enumerate(vol_slices.items()):
    slices.sort(key=lambda x:int(x.split("_slice")[-1].split(".npz")[0]))
    
    print(f"\n{'='*60}")
    print(f"Processing volume: {vid}")
    
    # Collect slices
    volume_imgs, volume_gts, volume_preds = [], [], []
    
    for s in tqdm(slices, desc="Processing slices"):
        npz = np.load(ROOT_PATH/"val_processed_224"/s)
        img_np, gt_np = npz["image"], npz["label"]

        # Get prediction
        t_img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            logits = net(t_img)
            pred_np = torch.argmax(torch.softmax(logits,1),1).cpu().numpy()[0]

        volume_imgs.append(img_np)
        volume_gts.append(gt_np)
        volume_preds.append(pred_np)

    # Convert to 3D volumes
    volume_imgs = np.stack(volume_imgs)
    volume_gts = np.stack(volume_gts)
    volume_preds_orig = np.stack(volume_preds)
    
    # Apply post-processing
    print("Applying post-processing...")
    volume_preds_basic = post_processor.process(volume_preds_orig > 0, method="basic")
    volume_preds_advanced = post_processor.process(volume_preds_orig > 0, method="advanced")
    
    # Calculate metrics
    results = {}
    for method, pred_vol in [
        ("original", volume_preds_orig > 0),
        ("basic", volume_preds_basic), 
        ("advanced", volume_preds_advanced)
    ]:
        dice = calculate_dice_score(pred_vol, volume_gts > 0)
        hd95 = calculate_hd95(pred_vol, volume_gts > 0)
        
        results[method] = {"dice": dice, "hd95": hd95}
        all_metrics[method]["dice"].append(dice)
        all_metrics[method]["hd95"].append(hd95)
        
        writer.add_scalar(f"dice_{method}", dice, v_idx)
        writer.add_scalar(f"hd95_{method}", hd95, v_idx)

    # Print results
    print("Volume metrics:")
    for method in methods:
        metrics = results[method]
        print(f"  {method:>10}: Dice {metrics['dice']:.4f}, HD95 {metrics['hd95']:.2f}")
    
    # Track best/worst (using advanced method)
    adv_dice = results["advanced"]["dice"]
    if adv_dice > best_dice:
        best_vol, best_dice = vid, adv_dice
    if adv_dice < worst_dice:
        worst_vol, worst_dice = vid, adv_dice

# ───────── Save Best and Worst Volumes ─────────────────────────────────────
def save_volume_comparison(vol_name, imgs, gts, pred_orig, pred_basic, pred_advanced):
    """Save volume comparison with all methods"""
    aff = np.eye(4)
    
    # Save all versions
    nib.save(nib.Nifti1Image(imgs.astype(np.float32), aff), 
             PRED_DIR / f"{vol_name}_img.nii.gz")
    nib.save(nib.Nifti1Image(gts.astype(np.uint8), aff),
             PRED_DIR / f"{vol_name}_gt.nii.gz") 
    nib.save(nib.Nifti1Image((pred_orig > 0).astype(np.uint8), aff),
             PRED_DIR / f"{vol_name}_original.nii.gz")
    nib.save(nib.Nifti1Image(pred_basic.astype(np.uint8), aff),
             PRED_DIR / f"{vol_name}_basic.nii.gz")
    nib.save(nib.Nifti1Image(pred_advanced.astype(np.uint8), aff),
             PRED_DIR / f"{vol_name}_advanced.nii.gz")

# Save best and worst volumes
for save_vol in [best_vol, worst_vol]:
    if save_vol:
        # Re-process this volume for saving
        vol_slices_to_save = vol_slices[save_vol]
        vol_slices_to_save.sort(key=lambda x:int(x.split("_slice")[-1].split(".npz")[0]))
        
        save_imgs, save_gts, save_preds = [], [], []
        for s in vol_slices_to_save:
            npz = np.load(ROOT_PATH/"val_processed_224"/s)
            img_np, gt_np = npz["image"], npz["label"]
            
            t_img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                logits = net(t_img)
                pred_np = torch.argmax(torch.softmax(logits,1),1).cpu().numpy()[0]
                
            save_imgs.append(img_np)
            save_gts.append(gt_np)
            save_preds.append(pred_np)
        
        save_imgs = np.stack(save_imgs)
        save_gts = np.stack(save_gts)
        save_preds_orig = np.stack(save_preds)
        
        save_preds_basic = post_processor.process(save_preds_orig > 0, method="basic")
        save_preds_advanced = post_processor.process(save_preds_orig > 0, method="advanced")
        
        save_volume_comparison(save_vol, save_imgs, save_gts, 
                             save_preds_orig, save_preds_basic, save_preds_advanced)

# ───────── Final Summary ─────────────────────────────────────────────────
print("\n" + "="*80)
print("POST-PROCESSING COMPARISON SUMMARY")
print("="*80)

for method in methods:
    dice_mean = np.mean(all_metrics[method]["dice"])
    dice_std = np.std(all_metrics[method]["dice"])
    hd95_mean = np.mean(all_metrics[method]["hd95"])
    hd95_std = np.std(all_metrics[method]["hd95"])
    
    print(f"{method.upper():>10}: Dice {dice_mean:.4f}±{dice_std:.4f}, "
          f"HD95 {hd95_mean:.2f}±{hd95_std:.2f}")

print("\nIMPROVEMENTS OVER ORIGINAL:")
orig_dice = np.mean(all_metrics["original"]["dice"])
orig_hd95 = np.mean(all_metrics["original"]["hd95"])

for method in ["basic", "advanced"]:
    method_dice = np.mean(all_metrics[method]["dice"])
    method_hd95 = np.mean(all_metrics[method]["hd95"])
    
    dice_imp = ((method_dice - orig_dice) / orig_dice) * 100
    hd95_imp = ((orig_hd95 - method_hd95) / orig_hd95) * 100
    
    print(f"{method.upper():>10}: Dice {dice_imp:+.1f}%, HD95 {hd95_imp:+.1f}%")

print(f"\nBest volume:  {best_vol} (Dice: {best_dice:.4f})")
print(f"Worst volume: {worst_vol} (Dice: {worst_dice:.4f})")
print(f"Results saved to: {PRED_DIR}")

writer.close()
