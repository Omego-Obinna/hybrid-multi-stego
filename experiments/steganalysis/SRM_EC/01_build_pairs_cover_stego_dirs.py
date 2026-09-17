#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IMG_EXTS = {".png", ".pgm", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def list_images_recursive(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
    )


def normalise_stem(path: Path):
    stem = path.stem
    for prefix in ["cover_", "stego_", "boss_", "BOSS_"]:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def choose_stego_search_dir(stego_root: Path, payload_code: str):
    candidates = [
        stego_root / f"payload_{payload_code}",
        stego_root / f"payload_{payload_code}" / "stego",
        stego_root / f"payload_{payload_code}" / "images",
        stego_root / f"payload_{payload_code}" / "stego_images",
        stego_root / payload_code,
        stego_root,
    ]

    for c in candidates:
        if c.exists() and list_images_recursive(c):
            return c

    raise FileNotFoundError(
        f"No stego images found for payload_{payload_code} under {stego_root}"
    )


def build_index(paths):
    index = {}
    duplicates = {}

    for p in paths:
        key = normalise_stem(p)
        if key in index:
            duplicates.setdefault(key, []).append(p)
        else:
            index[key] = p

    if duplicates:
        print(f"[WARN] Found {len(duplicates)} duplicate stems. First occurrence will be used.")

    return index


def assign_splits(pair_ids, seed, train_ratio, val_ratio):
    rng = np.random.default_rng(seed)
    pair_ids = np.array(pair_ids)
    rng.shuffle(pair_ids)

    n = len(pair_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train_ids = set(pair_ids[:n_train])
    val_ids = set(pair_ids[n_train:n_train + n_val])
    test_ids = set(pair_ids[n_train + n_val:])

    split_map = {}
    for x in train_ids:
        split_map[x] = "train"
    for x in val_ids:
        split_map[x] = "val"
    for x in test_ids:
        split_map[x] = "test"

    return split_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover-dir", required=True)
    ap.add_argument("--stego-root", required=True)
    ap.add_argument("--payload-code", required=True, help="Example: 010, 020, 030, 040")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for smoke testing")
    args = ap.parse_args()

    cover_dir = Path(args.cover_dir).expanduser().resolve()
    stego_root = Path(args.stego_root).expanduser().resolve()
    payload_code = args.payload_code

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must equal 1.0")

    stego_dir = choose_stego_search_dir(stego_root, payload_code)

    print(f"[INFO] Cover directory: {cover_dir}")
    print(f"[INFO] Stego search directory: {stego_dir}")

    cover_paths = list_images_recursive(cover_dir)
    stego_paths = list_images_recursive(stego_dir)

    print(f"[INFO] Found cover images: {len(cover_paths)}")
    print(f"[INFO] Found stego images: {len(stego_paths)}")

    cover_index = build_index(cover_paths)
    stego_index = build_index(stego_paths)

    matched_ids = sorted(set(cover_index.keys()) & set(stego_index.keys()))

    if args.limit and args.limit > 0:
        matched_ids = matched_ids[:args.limit]

    if not matched_ids:
        raise SystemExit(
            "No matching cover/stego image stems found. "
            "Check whether the BOSSbase files and stego files share matching names."
        )

    split_map = assign_splits(
        matched_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    rows = []
    for pair_id in matched_ids:
        rows.append(
            {
                "payload_code": payload_code,
                "split": split_map[pair_id],
                "pair_id": pair_id,
                "cover_path": str(cover_index[pair_id]),
                "stego_path": str(stego_index[pair_id]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "pair_id"]).reset_index(drop=True)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[OK] Wrote pairs to: {out_csv}")
    print(df.groupby("split").size())
    print(f"[OK] Total matched pairs: {len(df)}")


if __name__ == "__main__":
    main()
