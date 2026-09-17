#!/usr/bin/env python3
"""
01_prepare_matched_linkability_dataset.py

Validate a completed 12-gamma-pair matched-cover Ours-QIM run, extract
auditable text/image features, and create a balanced linked-vs-shuffled
dataset for empirical cross-channel linkability testing.

Positive: candidate gamma pair equals the true generating pair.
Negative: candidate gamma pair differs, while stego object, cover, and
payload are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

VERSION = "2026-07-17-matched-linkability-prepare-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a matched multi-gamma linkability dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--merged-log", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--payloads", default="0.10,0.20,0.40")
    p.add_argument("--expected-pairs", type=int, default=12)
    p.add_argument("--expected-covers", type=int, default=1000)
    p.add_argument("--n-negatives", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--image-workers", type=int, default=4)
    p.add_argument("--hash-dim", type=int, default=32)
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def bar(it: Iterable[Any], *, total: int | None, desc: str, unit: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(it, total=total, desc=desc, unit=unit, dynamic_ncols=True)
    return it


def parse_payloads(raw: str) -> list[float]:
    vals = sorted({round(float(x.strip()), 2) for x in raw.split(",") if x.strip()})
    if not vals:
        raise SystemExit("No payloads supplied.")
    return vals


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Merged log does not exist: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, convert_dates=False)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise SystemExit("Merged log must be JSONL/NDJSON or CSV.")


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = list(df.columns)
    resolved = {
        "pair": first_existing(cols, ["gamma_pair_id", "pair_id"]),
        "gamma1": first_existing(cols, ["gamma1_path", "gamma_1_path"]),
        "gamma2": first_existing(cols, ["gamma2_path", "gamma_2_path"]),
        "stego": first_existing(cols, ["stego_path", "output_path", "stego"]),
        "cover": first_existing(cols, ["original_cover_path", "cover_path", "input_path", "cover"]),
        "payload": first_existing(cols, ["payload_bpp", "payload", "target_payload", "target_bpp"]),
        "image_id": first_existing(cols, ["image_id", "cover_id"]),
    }
    missing = [k for k, v in resolved.items() if k != "image_id" and v is None]
    if missing:
        raise SystemExit(
            f"Could not resolve required columns {missing}. Available columns: {cols}"
        )
    return {k: v for k, v in resolved.items() if v is not None}


def norm_payload(value: Any) -> float:
    try:
        return round(float(value), 2)
    except Exception:
        s = str(value)
        m = re.search(r"0[._]?([124])00?", s)
        if m:
            return {"1": 0.10, "2": 0.20, "4": 0.40}[m.group(1)]
        raise ValueError(f"Cannot parse payload: {value!r}")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def read_text(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.read_text(encoding="utf-8", errors="replace")


def summary_text_features(prefix: str, text: str) -> dict[str, float]:
    lines = text.splitlines()
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    chars = list(text)
    line_lengths = np.array([len(x) for x in lines], dtype=float)
    word_lengths = np.array([len(x) for x in words], dtype=float)
    n_chars = max(len(chars), 1)
    n_words = max(len(words), 1)
    vocab = len(set(words))
    return {
        f"{prefix}lines": float(len(lines)),
        f"{prefix}chars": float(len(text)),
        f"{prefix}words": float(len(words)),
        f"{prefix}vocab": float(vocab),
        f"{prefix}type_token": float(vocab / n_words),
        f"{prefix}mean_line_len": float(line_lengths.mean()) if len(line_lengths) else 0.0,
        f"{prefix}std_line_len": float(line_lengths.std()) if len(line_lengths) else 0.0,
        f"{prefix}mean_word_len": float(word_lengths.mean()) if len(word_lengths) else 0.0,
        f"{prefix}std_word_len": float(word_lengths.std()) if len(word_lengths) else 0.0,
        f"{prefix}alpha_ratio": sum(c.isalpha() for c in chars) / n_chars,
        f"{prefix}digit_ratio": sum(c.isdigit() for c in chars) / n_chars,
        f"{prefix}space_ratio": sum(c.isspace() for c in chars) / n_chars,
        f"{prefix}punct_ratio": sum((not c.isalnum()) and (not c.isspace()) for c in chars) / n_chars,
        f"{prefix}upper_ratio": sum(c.isupper() for c in chars) / n_chars,
    }


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = float(np.linalg.norm(x))
    return x / norm if norm > 0 else np.zeros_like(x)


def gamma_features(pair_id: str, gamma1_path: str, gamma2_path: str, hash_dim: int) -> dict[str, Any]:
    g1 = read_text(gamma1_path)
    g2 = read_text(gamma2_path)
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=hash_dim,
        alternate_sign=True,
        norm="l2",
        lowercase=True,
    )
    matrix = vectorizer.transform([g1, g2]).toarray().astype(np.float64)
    g1_vec, g2_vec = matrix[0], matrix[1]
    mean_vec = l2_normalize((g1_vec + g2_vec) / 2.0)
    diff_vec = np.abs(g1_vec - g2_vec)
    row: dict[str, Any] = {
        "candidate_gamma_pair_id": pair_id,
        "gamma1_path": gamma1_path,
        "gamma2_path": gamma2_path,
        "gamma1_sha256": hashlib.sha256(g1.encode("utf-8")).hexdigest(),
        "gamma2_sha256": hashlib.sha256(g2.encode("utf-8")).hexdigest(),
        "gamma_pair_cosine": float(np.dot(l2_normalize(g1_vec), l2_normalize(g2_vec))),
    }
    row.update(summary_text_features("gamma_g1_", g1))
    row.update(summary_text_features("gamma_g2_", g2))
    for i in range(hash_dim):
        row[f"gamma_g1_h{i:02d}"] = float(g1_vec[i])
        row[f"gamma_g2_h{i:02d}"] = float(g2_vec[i])
        row[f"gamma_sig_{i:02d}"] = float(mean_vec[i])
        row[f"gamma_diff_h{i:02d}"] = float(diff_vec[i])
    return row


def entropy_from_hist(hist: np.ndarray) -> float:
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p /= total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def moment_stats(x: np.ndarray) -> tuple[float, float]:
    flat = x.astype(np.float64).ravel()
    mean = float(flat.mean())
    std = float(flat.std())
    if std <= 1e-12:
        return 0.0, 0.0
    z = (flat - mean) / std
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


def image_features(path_str: str, sig_dim: int) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0

    flat = arr.ravel()
    skew, kurt = moment_stats(flat)
    dx = np.diff(arr, axis=1)
    dy = np.diff(arr, axis=0)
    ddiag = arr[1:, 1:] - arr[:-1, :-1]
    center = arr[1:-1, 1:-1]
    hp = center - 0.25 * (
        arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:]
    )

    half = sig_dim // 2
    pix_hist, _ = np.histogram(flat, bins=half, range=(0.0, 1.0))
    hp_abs = np.clip(np.abs(hp).ravel(), 0.0, 1.0)
    res_hist, _ = np.histogram(hp_abs, bins=sig_dim - half, range=(0.0, 1.0))
    signature = l2_normalize(np.concatenate([pix_hist, res_hist]).astype(np.float64))

    row: dict[str, Any] = {
        "stego_path": str(path),
        "img_width": float(arr.shape[1]),
        "img_height": float(arr.shape[0]),
        "img_mean": float(flat.mean()),
        "img_std": float(flat.std()),
        "img_min": float(flat.min()),
        "img_max": float(flat.max()),
        "img_q01": float(np.quantile(flat, 0.01)),
        "img_q05": float(np.quantile(flat, 0.05)),
        "img_q25": float(np.quantile(flat, 0.25)),
        "img_q50": float(np.quantile(flat, 0.50)),
        "img_q75": float(np.quantile(flat, 0.75)),
        "img_q95": float(np.quantile(flat, 0.95)),
        "img_q99": float(np.quantile(flat, 0.99)),
        "img_skew": skew,
        "img_kurtosis": kurt,
        "img_entropy": entropy_from_hist(pix_hist),
        "img_dx_mean_abs": float(np.mean(np.abs(dx))),
        "img_dx_std": float(dx.std()),
        "img_dy_mean_abs": float(np.mean(np.abs(dy))),
        "img_dy_std": float(dy.std()),
        "img_diag_mean_abs": float(np.mean(np.abs(ddiag))),
        "img_diag_std": float(ddiag.std()),
        "img_hp_mean_abs": float(np.mean(np.abs(hp))),
        "img_hp_std": float(hp.std()),
        "img_hp_q50": float(np.quantile(hp_abs, 0.50)),
        "img_hp_q90": float(np.quantile(hp_abs, 0.90)),
        "img_hp_q99": float(np.quantile(hp_abs, 0.99)),
    }
    for i, value in enumerate(signature):
        row[f"img_sig_{i:02d}"] = float(value)
    return row


def cross_features(image_row: dict[str, Any], gamma_row: dict[str, Any], dim: int) -> dict[str, float]:
    iv = np.array([image_row[f"img_sig_{i:02d}"] for i in range(dim)], dtype=float)
    gv = np.array([gamma_row[f"gamma_sig_{i:02d}"] for i in range(dim)], dtype=float)
    ivn, gvn = l2_normalize(iv), l2_normalize(gv)
    diff = np.abs(ivn - gvn)
    prod = ivn * gvn
    out = {
        "cross_cosine": float(np.dot(ivn, gvn)),
        "cross_l1": float(np.sum(diff)),
        "cross_l2": float(np.linalg.norm(ivn - gvn)),
        "cross_dot": float(np.dot(iv, gv)),
    }
    for i in range(dim):
        out[f"cross_abs_{i:02d}"] = float(diff[i])
        out[f"cross_prod_{i:02d}"] = float(prod[i])
    return out


def main() -> None:
    args = parse_args()
    print(f"[INFO] Version: {VERSION}")
    payloads = parse_payloads(args.payloads)
    if args.n_negatives < 1:
        raise SystemExit("--n-negatives must be at least 1.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_log(args.merged_log)
    cols = resolve_columns(df)
    print("[INFO] Resolved columns:", cols)

    sessions = pd.DataFrame({
        "true_gamma_pair_id": df[cols["pair"]].astype(str),
        "gamma1_path": df[cols["gamma1"]].astype(str),
        "gamma2_path": df[cols["gamma2"]].astype(str),
        "stego_path": df[cols["stego"]].astype(str),
        "original_cover_path": df[cols["cover"]].astype(str),
        "payload_bpp": df[cols["payload"]].map(norm_payload),
    })
    sessions["image_id"] = (
        df[cols["image_id"]].astype(str)
        if "image_id" in cols
        else sessions["original_cover_path"].map(lambda x: Path(x).stem)
    )
    sessions = sessions[sessions["payload_bpp"].isin(payloads)].copy()
    sessions["payload_code"] = sessions["payload_bpp"].map(lambda x: f"{int(round(x*100)):03d}")
    sessions["cover_id"] = sessions["original_cover_path"].map(
        lambda x: sha1_text(str(Path(x).resolve(strict=False)))
    )
    sessions["stego_id"] = sessions.apply(
        lambda r: sha1_text(
            f"{r.true_gamma_pair_id}|{r.payload_bpp:.2f}|{r.original_cover_path}|{r.stego_path}"
        ),
        axis=1,
    )

    valid = np.ones(len(sessions), dtype=bool)
    for path_col in ["gamma1_path", "gamma2_path", "stego_path", "original_cover_path"]:
        exists = sessions[path_col].map(lambda x: Path(x).is_file()).to_numpy()
        if not exists.all():
            count = int((~exists).sum())
            if args.strict:
                examples = sessions.loc[~exists, path_col].head(10).tolist()
                raise SystemExit(f"{count} missing files in {path_col}; examples: {examples}")
            print(f"[WARN] Dropping {count} rows with missing {path_col}.")
            valid &= exists
    sessions = sessions.loc[valid].copy()

    duplicate_key = ["true_gamma_pair_id", "payload_bpp", "cover_id"]
    duplicates = int(sessions.duplicated(duplicate_key).sum())
    if duplicates:
        raise SystemExit(f"Found {duplicates} duplicate pair/payload/cover records.")

    pair_ids = sorted(sessions["true_gamma_pair_id"].unique())
    if len(pair_ids) != args.expected_pairs:
        raise SystemExit(f"Expected {args.expected_pairs} gamma pairs, found {len(pair_ids)}: {pair_ids}")

    expected_cells = len(pair_ids) * len(payloads)
    cover_cell_counts = (
        sessions[["cover_id", "true_gamma_pair_id", "payload_bpp"]]
        .drop_duplicates()
        .groupby("cover_id")
        .size()
    )
    complete_covers = set(cover_cell_counts[cover_cell_counts == expected_cells].index)
    incomplete = sessions["cover_id"].nunique() - len(complete_covers)
    if incomplete:
        print(f"[WARN] Removing {incomplete} covers without all {expected_cells} matched cells.")
    sessions = sessions[sessions["cover_id"].isin(complete_covers)].copy()

    if args.strict and sessions["cover_id"].nunique() != args.expected_covers:
        raise SystemExit(
            f"Expected {args.expected_covers} complete matched covers, "
            f"found {sessions['cover_id'].nunique()}."
        )

    cell_counts = sessions.groupby(["payload_bpp", "true_gamma_pair_id"]).size().unstack(fill_value=0)
    print("\n[INFO] Matched cell counts:")
    print(cell_counts.to_string())
    print(
        f"\n[INFO] Sessions={len(sessions)}, pairs={len(pair_ids)}, "
        f"covers={sessions['cover_id'].nunique()}, payloads={payloads}"
    )

    pair_meta = (
        sessions[["true_gamma_pair_id", "gamma1_path", "gamma2_path"]]
        .drop_duplicates()
        .sort_values("true_gamma_pair_id")
    )
    if pair_meta["true_gamma_pair_id"].duplicated().any():
        raise SystemExit("A gamma pair ID maps to multiple gamma path combinations.")

    gamma_rows = []
    for row in bar(
        pair_meta.itertuples(index=False),
        total=len(pair_meta),
        desc="extract gamma features",
        unit="pair",
        enabled=args.progress,
    ):
        gamma_rows.append(
            gamma_features(
                row.true_gamma_pair_id,
                row.gamma1_path,
                row.gamma2_path,
                args.hash_dim,
            )
        )
    gamma_df = pd.DataFrame(gamma_rows)
    gamma_map = {r["candidate_gamma_pair_id"]: r for r in gamma_df.to_dict(orient="records")}

    unique_images = sessions[["stego_id", "stego_path"]].drop_duplicates()
    items = list(unique_images.itertuples(index=False, name=None))

    def extract_one(item: tuple[str, str]) -> dict[str, Any]:
        stego_id, path = item
        result = image_features(path, args.hash_dim)
        result["stego_id"] = stego_id
        return result

    image_rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.image_workers)) as executor:
        iterator = executor.map(extract_one, items)
        for row in bar(
            iterator,
            total=len(items),
            desc=f"extract image features ({max(1,args.image_workers)} workers)",
            unit="image",
            enabled=args.progress,
        ):
            image_rows.append(row)
    image_df = pd.DataFrame(image_rows)
    image_map = {r["stego_id"]: r for r in image_df.to_dict(orient="records")}

    pair_index = {pair: i for i, pair in enumerate(pair_ids)}
    dataset_rows: list[dict[str, Any]] = []
    session_records = sessions.sort_values(
        ["payload_bpp", "cover_id", "true_gamma_pair_id"]
    ).to_dict(orient="records")

    max_neg = min(args.n_negatives, len(pair_ids) - 1)
    for session in bar(
        session_records,
        total=len(session_records),
        desc="build linked/shuffled tuples",
        unit="session",
        enabled=args.progress,
    ):
        true_pair = session["true_gamma_pair_id"]
        true_index = pair_index[true_pair]
        image_row = image_map[session["stego_id"]]
        candidates = [(true_pair, 1)]
        for offset in range(1, max_neg + 1):
            candidates.append((pair_ids[(true_index + offset) % len(pair_ids)], 0))

        for candidate_pair, label in candidates:
            gamma_row = gamma_map[candidate_pair]
            out: dict[str, Any] = {
                "label": int(label),
                "payload_bpp": float(session["payload_bpp"]),
                "payload_code": session["payload_code"],
                "cover_id": session["cover_id"],
                "image_id": session["image_id"],
                "stego_id": session["stego_id"],
                "original_cover_path": session["original_cover_path"],
                "stego_path": session["stego_path"],
                "true_gamma_pair_id": true_pair,
                "candidate_gamma_pair_id": candidate_pair,
            }
            out.update({k: v for k, v in image_row.items() if k.startswith("img_")})
            out.update({k: v for k, v in gamma_row.items() if k.startswith("gamma_")})
            out.update(cross_features(image_row, gamma_row, args.hash_dim))
            dataset_rows.append(out)

    dataset = pd.DataFrame(dataset_rows)
    label_counts = dataset.groupby(["payload_code", "label"]).size()
    print("\n[INFO] Binary label counts:")
    print(label_counts.to_string())

    for code, group in dataset.groupby("payload_code"):
        if set(group["label"].astype(int).unique()) != {0, 1}:
            raise RuntimeError(f"Payload {code} lacks both labels.")

    sessions_path = args.out_dir / "linkability_sessions.csv"
    gamma_path = args.out_dir / "gamma_pair_features.csv"
    image_path = args.out_dir / "image_features.csv.gz"
    dataset_path = args.out_dir / "linkability_binary_dataset.csv.gz"
    sessions.to_csv(sessions_path, index=False)
    gamma_df.to_csv(gamma_path, index=False)
    image_df.to_csv(image_path, index=False, compression="gzip")
    dataset.to_csv(dataset_path, index=False, compression="gzip")

    summary = {
        "version": VERSION,
        "merged_log": str(args.merged_log.resolve()),
        "payloads": payloads,
        "n_gamma_pairs": len(pair_ids),
        "gamma_pair_ids": pair_ids,
        "n_complete_covers": int(sessions["cover_id"].nunique()),
        "n_sessions": int(len(sessions)),
        "n_binary_rows": int(len(dataset)),
        "n_negatives_per_positive": max_neg,
        "hash_dim": args.hash_dim,
        "feature_counts": {
            "image": len([c for c in dataset if c.startswith("img_")]),
            "gamma": len([c for c in dataset if c.startswith("gamma_")]),
            "cross": len([c for c in dataset if c.startswith("cross_")]),
        },
    }
    summary_path = args.out_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[OK] Wrote:")
    for path in [sessions_path, gamma_path, image_path, dataset_path, summary_path]:
        print("  ", path)


if __name__ == "__main__":
    main()
