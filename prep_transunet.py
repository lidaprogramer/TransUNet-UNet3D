#!/usr/bin/env python3
"""
prep_transunet.py ─ organise ImageCas volumes for TransUNet

Usage examples
--------------
# one folder
python prep_transunet.py \
       --src ~/hist/imagecas/1-200 \
       --dst ~/hist/TransUNet/data

# many folders at once (you can repeat --src)
python prep_transunet.py \
       --src ~/hist/imagecas/1-200 \
       --src ~/hist/imagecas/extracted_201-400 \
       --src ~/hist/imagecas/extracted_401-600 \
       --dst ~/hist/TransUNet/data \
       --copy
"""

import argparse
from pathlib import Path
import shutil
import sys


def safe_move_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    """Move (or copy) *src* to *dst* without overwriting existing files."""
    if dst.exists():
        print(f"⚠️  skip {dst.name} (already exists)", file=sys.stderr)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def preprocess(source_dirs, dst_root: Path, *, copy: bool) -> None:
    train_dir = dst_root / "train"
    label_dir = dst_root / "label"
    train_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for src in source_dirs:
        src = src.expanduser().resolve()
        prefix = src.name               # e.g. "201-400"
        if not src.is_dir():
            print(f"✖ {src} is not a directory – skipping", file=sys.stderr)
            continue

        print(f"▶ processing {src}")
        for f in src.rglob("*.nii.gz"):
            base = f.name               # e.g. "091.img.nii.gz"
            new_name = f"{prefix}_{base}"
            if ".img." in base:
                target = train_dir / new_name
            elif ".label." in base:
                target = label_dir / new_name
            else:
                continue
            safe_move_or_copy(f, target, copy=copy)

    print("✅ done.  Images in", train_dir)
    print("          Labels in", label_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare ImageCas data for TransUNet")
    ap.add_argument("--src", type=Path, action="append", required=True,
                    help="One source directory per --src (repeatable).")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Destination root (will create train/ and label/).")
    ap.add_argument("--copy", action="store_true",
                    help="Copy instead of move (keeps originals).")
    args = ap.parse_args()

    preprocess(args.src, args.dst, copy=args.copy)
