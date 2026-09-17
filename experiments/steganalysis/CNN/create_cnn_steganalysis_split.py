#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_yenet_main_split.py

Create paired cover/stego train/val/test folders for the main Ye-Net detection
experiment on the full BOSSBase_1.01 run.

Default split:
  70% train, 10% validation, 20% test

Important rule:
  A cover image and its matching stego image are always assigned to the same split.
  The same image IDs are also assigned to the same split across all payloads.

For storage efficiency, the default operation is symlink rather than copy.
Use --mode copy only if your training code cannot follow symlinks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".pgm", ".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

DEFAULT_COVER_ROOT = Path("/home/omego/Downloads/BOSSbase/BOSSbase_1.01")
DEFAULT_STEGO_ROOT = Path("/home/omego/Documents/Hybrid_Stego_Eva/ours_qim_v3_2_fused075_payloads_FULL")
DEFAULT_OUT_ROOT = Path("/home/omego/Documents/Hybrid_Stego_Eva/yenet_main_split_v3_2_fused075")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_images(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths.sort(key=lambda p: str(p))

    mapping: Dict[str, Path] = {}
    duplicates: List[str] = []
    for p in paths:
        if p.name in mapping:
            duplicates.append(p.name)
        mapping[p.name] = p

    if duplicates:
        raise RuntimeError(
            f"Duplicate filenames found under {root}. Examples: {duplicates[:10]}"
        )
    return mapping


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def make_ratio_split(names: List[str], seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, List[str]]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    train_ratio /= total
    val_ratio /= total

    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:n_train + n_val + n_test]),
    }


def write_manifest(path: Path, rows: List[dict], include_hash: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "payload_label",
        "payload_bpp",
        "split",
        "filename",
        "cover_path",
        "stego_path",
    ]
    if include_hash:
        fieldnames += ["cover_sha256", "stego_sha256"]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create full paired Ye-Net train/val/test split for Ours-QIM-v3.2-Fused-CF.")
    p.add_argument("--cover-root", type=Path, default=DEFAULT_COVER_ROOT)
    p.add_argument("--stego-root", type=Path, default=DEFAULT_STEGO_ROOT)
    p.add_argument("--stego-010", type=Path, default=None)
    p.add_argument("--stego-020", type=Path, default=None)
    p.add_argument("--stego-040", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)

    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--max-pairs", type=int, default=0, help="Optional cap for debugging. 0 means use all common pairs.")
    p.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    p.add_argument("--hash", action="store_true", help="Compute SHA-256 hashes in manifests. Slower on full BOSSBase.")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output folder.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.out_root.exists() and not args.overwrite:
        raise RuntimeError(
            f"Output directory already exists: {args.out_root}\n"
            f"Use --overwrite if you intentionally want to refresh symlinks/manifests."
        )

    stego_dirs = {
        "010": args.stego_010 or (args.stego_root / "ours_qim_0.100"),
        "020": args.stego_020 or (args.stego_root / "ours_qim_0.200"),
        "040": args.stego_040 or (args.stego_root / "ours_qim_0.400"),
    }
    payload_bpp = {"010": "0.10", "020": "0.20", "040": "0.40"}

    cover_map = find_images(args.cover_root)
    stego_maps = {label: find_images(path) for label, path in stego_dirs.items()}

    common = set(cover_map)
    for smap in stego_maps.values():
        common &= set(smap)

    common_names = sorted(common, key=lambda x: (len(Path(x).stem), Path(x).stem, x))
    if args.max_pairs and args.max_pairs > 0:
        common_names = common_names[:args.max_pairs]

    if not common_names:
        raise RuntimeError("No common cover/stego filenames found.")

    split = make_ratio_split(
        common_names,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []

    print(f"[INFO] cover_root: {args.cover_root}")
    print(f"[INFO] stego_root: {args.stego_root}")
    print(f"[INFO] out_root:   {args.out_root}")
    print(f"[INFO] operation:  {args.mode}")
    print(f"[INFO] common paired image IDs: {len(common_names)}")
    print(f"[INFO] split sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

    split_info_path = args.out_root / "split_info.json"
    split_info_path.write_text(
        "{\n"
        + f'  "cover_root": "{args.cover_root}",\n'
        + f'  "stego_root": "{args.stego_root}",\n'
        + f'  "out_root": "{args.out_root}",\n'
        + f'  "seed": {args.seed},\n'
        + f'  "train_ratio": {args.train_ratio},\n'
        + f'  "val_ratio": {args.val_ratio},\n'
        + f'  "test_ratio": {args.test_ratio},\n'
        + f'  "n_common_pairs": {len(common_names)},\n'
        + f'  "n_train": {len(split["train"])},\n'
        + f'  "n_val": {len(split["val"])},\n'
        + f'  "n_test": {len(split["test"])},\n'
        + f'  "mode": "{args.mode}"\n'
        + "}\n"
    )

    for payload_label, stego_map in stego_maps.items():
        payload_root = args.out_root / f"payload_{payload_label}"
        payload_rows: List[dict] = []

        for split_name, filenames in split.items():
            for subdir in ["cover", "stego"]:
                (payload_root / split_name / subdir).mkdir(parents=True, exist_ok=True)

            for filename in filenames:
                cover_src = cover_map[filename]
                stego_src = stego_map[filename]
                cover_dst = payload_root / split_name / "cover" / filename
                stego_dst = payload_root / split_name / "stego" / filename

                copy_or_link(cover_src, cover_dst, args.mode)
                copy_or_link(stego_src, stego_dst, args.mode)

                row = {
                    "payload_label": payload_label,
                    "payload_bpp": payload_bpp[payload_label],
                    "split": split_name,
                    "filename": filename,
                    "cover_path": str(cover_dst),
                    "stego_path": str(stego_dst),
                }
                if args.hash:
                    row["cover_sha256"] = sha256_file(cover_src)
                    row["stego_sha256"] = sha256_file(stego_src)

                payload_rows.append(row)
                all_rows.append(row)

        write_manifest(payload_root / "split_manifest.csv", payload_rows, include_hash=args.hash)
        print(f"[OK] payload_{payload_label}: {payload_root}")

    write_manifest(args.out_root / "split_manifest_all_payloads.csv", all_rows, include_hash=args.hash)
    print(f"[DONE] Manifest: {args.out_root / 'split_manifest_all_payloads.csv'}")
    print(f"[DONE] Split info: {split_info_path}")


if __name__ == "__main__":
    main()
