#!/usr/bin/env python3
"""
03_rank_linkability_candidates.py

For each held-out stego object, score every candidate gamma pair and rank
its true generating pair. Train/test separation is grouped by original cover.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

VERSION = "2026-07-17-matched-linkability-ranking-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank candidate gamma pairs for held-out stego objects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--view",
        choices=["text_image", "cross_only", "text_only", "image_only"],
        default="text_image",
    )
    p.add_argument(
        "--classifier",
        choices=["logistic", "extra_trees"],
        default="extra_trees",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--batch-images", type=int, default=256)
    return p.parse_args()


def columns_for_view(df: pd.DataFrame, view: str) -> list[str]:
    img = [c for c in df if c.startswith("img_")]
    gamma = [c for c in df if c.startswith("gamma_")]
    cross = [c for c in df if c.startswith("cross_")]
    return {
        "image_only": img,
        "text_only": gamma,
        "cross_only": cross,
        "text_image": img + gamma + cross,
    }[view]


def build_model(name: str, seed: int, n_estimators: int, n_jobs: int):
    if name == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )),
    ])


def score_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    decision = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -50, 50)))


def l2(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else np.zeros_like(x)


def add_cross(row: dict[str, Any], dim: int) -> None:
    iv = np.array([row[f"img_sig_{i:02d}"] for i in range(dim)], dtype=float)
    gv = np.array([row[f"gamma_sig_{i:02d}"] for i in range(dim)], dtype=float)
    ivn, gvn = l2(iv), l2(gv)
    diff = np.abs(ivn - gvn)
    prod = ivn * gvn
    row["cross_cosine"] = float(np.dot(ivn, gvn))
    row["cross_l1"] = float(np.sum(diff))
    row["cross_l2"] = float(np.linalg.norm(ivn - gvn))
    row["cross_dot"] = float(np.dot(iv, gv))
    for i in range(dim):
        row[f"cross_abs_{i:02d}"] = float(diff[i])
        row[f"cross_prod_{i:02d}"] = float(prod[i])


def rank_metrics(ranks: np.ndarray, n_candidates: int) -> dict[str, float]:
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= min(3, n_candidates))),
        "top5": float(np.mean(ranks <= min(5, n_candidates))),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "random_top1": float(1.0 / n_candidates),
        "random_mrr": float(np.mean([1.0 / r for r in range(1, n_candidates + 1)])),
    }


def main() -> None:
    args = parse_args()
    print(f"[INFO] Version: {VERSION}")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset does not exist: {args.dataset}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset, low_memory=False)
    positive = df[df["label"].astype(int) == 1].drop_duplicates("stego_id").copy()
    pair_ids = sorted(df["candidate_gamma_pair_id"].astype(str).unique())
    n_candidates = len(pair_ids)
    if n_candidates < 2:
        raise SystemExit("Ranking requires at least two gamma pairs.")

    gamma_cols = [c for c in df if c.startswith("gamma_")]
    gamma_proto = (
        df[["candidate_gamma_pair_id"] + gamma_cols]
        .drop_duplicates("candidate_gamma_pair_id")
        .set_index("candidate_gamma_pair_id")
    )
    if len(gamma_proto) != n_candidates:
        raise SystemExit("Could not resolve one gamma feature vector per pair.")

    dim = len([c for c in df if c.startswith("img_sig_")])
    if dim == 0:
        raise SystemExit("No img_sig_* columns found.")
    feature_cols = columns_for_view(df, args.view)

    payloads = sorted(pd.to_numeric(positive["payload_bpp"]).round(2).unique())
    per_split: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    jobs = [(payload, split_id) for payload in payloads for split_id in range(1, args.n_splits + 1)]
    iterator = tqdm(jobs, desc="ranking experiments", unit="job") if tqdm else jobs

    for payload, split_id in iterator:
        binary_payload = df[np.isclose(pd.to_numeric(df["payload_bpp"]), payload)].copy()
        items = positive[np.isclose(pd.to_numeric(positive["payload_bpp"]), payload)].copy()

        unique_covers = items[["cover_id"]].drop_duplicates().reset_index(drop=True)
        splitter = GroupShuffleSplit(
            n_splits=args.n_splits,
            test_size=args.test_size,
            random_state=args.seed,
        )
        splits = list(splitter.split(unique_covers, groups=unique_covers["cover_id"]))
        _, test_cover_idx = splits[split_id - 1]
        test_covers = set(unique_covers.iloc[test_cover_idx]["cover_id"].astype(str))
        train_mask = ~binary_payload["cover_id"].astype(str).isin(test_covers)
        test_items = items[items["cover_id"].astype(str).isin(test_covers)].copy()

        model = build_model(
            args.classifier,
            args.seed + split_id,
            args.n_estimators,
            args.n_jobs,
        )
        model.fit(
            binary_payload.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan),
            binary_payload.loc[train_mask, "label"].astype(int),
        )

        ranks: list[int] = []
        test_records = test_items.to_dict(orient="records")
        img_cols = [c for c in df if c.startswith("img_")]

        for start in range(0, len(test_records), args.batch_images):
            chunk = test_records[start:start + args.batch_images]
            candidate_rows: list[dict[str, Any]] = []
            metadata: list[tuple[str, str, str, str]] = []

            for item in chunk:
                image_values = {c: item[c] for c in img_cols}
                for candidate in pair_ids:
                    row = dict(image_values)
                    row.update(gamma_proto.loc[candidate].to_dict())
                    add_cross(row, dim)
                    candidate_rows.append(row)
                    metadata.append((
                        item["stego_id"],
                        item["cover_id"],
                        item["true_gamma_pair_id"],
                        candidate,
                    ))

            candidate_df = pd.DataFrame(candidate_rows)
            scores = score_model(
                model,
                candidate_df[feature_cols].replace([np.inf, -np.inf], np.nan),
            )

            grouped: dict[str, list[tuple[str, float, str, str]]] = {}
            for meta, score in zip(metadata, scores):
                stego_id, cover_id, true_pair, candidate = meta
                grouped.setdefault(stego_id, []).append(
                    (candidate, float(score), cover_id, true_pair)
                )

            for stego_id, entries in grouped.items():
                entries.sort(key=lambda x: (-x[1], x[0]))
                true_pair = entries[0][3]
                rank = next(i for i, e in enumerate(entries, start=1) if e[0] == true_pair)
                ranks.append(rank)
                true_score = next(e[1] for e in entries if e[0] == true_pair)
                detail_rows.append({
                    "payload_bpp": float(payload),
                    "payload_code": f"{int(round(payload*100)):03d}",
                    "split_id": split_id,
                    "stego_id": stego_id,
                    "cover_id": entries[0][2],
                    "true_gamma_pair_id": true_pair,
                    "true_rank": rank,
                    "true_score": true_score,
                    "n_candidates": n_candidates,
                    "classifier": args.classifier,
                    "view": args.view,
                })

        rank_array = np.asarray(ranks, dtype=float)
        result = rank_metrics(rank_array, n_candidates)
        result.update({
            "payload_bpp": float(payload),
            "payload_code": f"{int(round(payload*100)):03d}",
            "split_id": split_id,
            "classifier": args.classifier,
            "view": args.view,
            "n_test_items": int(len(rank_array)),
            "n_candidates": n_candidates,
            "n_test_covers": len(test_covers),
        })
        per_split.append(result)

    per_split_df = pd.DataFrame(per_split)
    per_split_path = args.out_dir / "ranking_metrics_per_split.csv"
    per_split_df.to_csv(per_split_path, index=False)

    metric_cols = [
        "top1", "top3", "top5", "mrr", "mean_rank", "median_rank",
        "random_top1", "random_mrr",
    ]
    summary = (
        per_split_df
        .groupby(["payload_bpp", "payload_code", "classifier", "view"], as_index=False)
        .agg(
            **{f"{m}_mean": (m, "mean") for m in metric_cols},
            **{f"{m}_std": (m, "std") for m in metric_cols},
            n_splits=("split_id", "count"),
            n_candidates=("n_candidates", "first"),
        )
    )
    summary_path = args.out_dir / "ranking_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    detail_path = args.out_dir / "ranking_details.csv.gz"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, compression="gzip")

    config_path = args.out_dir / "ranking_config.json"
    config_path.write_text(json.dumps({
        "version": VERSION,
        "dataset": str(args.dataset.resolve()),
        "classifier": args.classifier,
        "view": args.view,
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_candidates": n_candidates,
    }, indent=2), encoding="utf-8")

    print("\n[OK] Wrote:")
    for path in [per_split_path, summary_path, detail_path, config_path]:
        print("  ", path)


if __name__ == "__main__":
    main()
