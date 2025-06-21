#!/usr/bin/env python3
"""preprocess_imagecas.py – slice 3‑D NIfTI volumes to 2‑D for TransUNet

This version autodetects the ImageCas naming scheme (`NN.img.nii.gz` ↔
`NN.label.nii.gz`) and no longer assumes *.mha files.

python preprocess_imagecas.py     --image-dir ~/hist/TransUNet/data/train     --label-dir ~/hist/TransUNet/data/label     --out-train
  ~/hist/TransUNet/data/train_processed_224     --out-test   ~/hist/TransUNet/data/val_processed_224     --lists-dir  ~/hist/TransUNet/lists/lists_ImageCas
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def resize_and_pad(arr: np.ndarray, size=(224, 224), pad_val=0):
    img = sitk.GetImageFromArray(arr)
    orig_size = img.GetSize()
    orig_spacing = img.GetSpacing()
    new_spacing = (
        orig_spacing[0] * orig_size[0] / size[0],
        orig_spacing[1] * orig_size[1] / size[1],
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    out = sitk.GetArrayFromImage(resampler.Execute(img))
    dh = max(0, size[0] - out.shape[0])
    dw = max(0, size[1] - out.shape[1])
    if dh or dw:
        out = np.pad(out, ((0, dh), (0, dw)), "constant", constant_values=pad_val)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# Pair collection
# ----------------------------------------------------------------------------


def collect_pairs(img_dir: Path, lbl_dir: Path):
    pairs = []
    for img in sorted(img_dir.glob("*.img.nii.gz")):
        lbl = lbl_dir / img.name.replace(".img.nii.gz", ".label.nii.gz")
        if lbl.exists():
            pairs.append((img, lbl))
    if not pairs:
        for img in sorted(img_dir.glob("*.nii.gz")):
            lbl = lbl_dir / img.name
            if lbl.exists():
                pairs.append((img, lbl))
    if not pairs:
        for img in sorted(img_dir.glob("*.mha")):
            lbl = lbl_dir / img.name
            if lbl.exists():
                pairs.append((img, lbl))
    if not pairs:
        raise RuntimeError(
            f"No matching pairs found in {img_dir} and {lbl_dir} – check names.")
    return list(zip(*pairs))  # imgs, lbls


# ----------------------------------------------------------------------------
# Volume slicing
# ----------------------------------------------------------------------------


def slice_volume(img_p: Path, lbl_p: Path, *, out_dir: Path, fh, size=(224, 224)):
    vol = sitk.ReadImage(str(img_p))
    lbl = sitk.ReadImage(str(lbl_p))
    v_arr = sitk.GetArrayFromImage(vol)
    l_arr = sitk.GetArrayFromImage(lbl)
    v_arr = np.clip(v_arr, 0, 4000)
    v_arr = (v_arr - v_arr.min()) / (v_arr.max() - v_arr.min() + 1e-7)
    for z in range(v_arr.shape[0]):
        img_sl = resize_and_pad(v_arr[z], size=size)
        lbl_sl = resize_and_pad(l_arr[z], size=size)
        name = f"{img_p.stem}_slice{z:03d}.npz"
        np.savez_compressed(out_dir / name, image=img_sl, label=lbl_sl)
        fh.write(name + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Slice ImageCas volumes for TransUNet")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-test", required=True)
    ap.add_argument("--lists-dir", required=True)
    ap.add_argument("--split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir = _expand(args.image_dir)
    lbl_dir = _expand(args.label_dir)
    out_tr = _expand(args.out_train)
    out_te = _expand(args.out_test)
    lists = _expand(args.lists_dir)

    rng = np.random.default_rng(args.seed)
    imgs, lbls = collect_pairs(img_dir, lbl_dir)
    order = rng.permutation(len(imgs))
    n_test = max(1, int(len(order) * args.split)) if len(order) else 0
    test_idx, train_idx = order[:n_test], order[n_test:]

    out_tr.mkdir(parents=True, exist_ok=True)
    out_te.mkdir(parents=True, exist_ok=True)
    lists.mkdir(parents=True, exist_ok=True)

    print(f"📦 {len(imgs)} volumes → train {len(train_idx)}, val {len(test_idx)}")

    with open(lists / "train.txt", "w") as ft:
        for i in train_idx:
            slice_volume(imgs[i], lbls[i], out_dir=out_tr, fh=ft)
    with open(lists / "test_vol.txt", "w") as fv:
        for i in test_idx:
            slice_volume(imgs[i], lbls[i], out_dir=out_te, fh=fv)
    with open(lists / "all.lst", "w") as fa:
        for p in imgs:
            fa.write(p.name + "\n")

    print("✅ Done – slices in", out_tr, "and", out_te)


if __name__ == "__main__":
    main()
