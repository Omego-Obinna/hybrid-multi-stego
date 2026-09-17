#!/usr/bin/env python3
"""
02_train_eval_linkability_warden.py

Train and evaluate simple empirical linkability wardens with repeated,
cover-grouped train/test splits.

Views:
  image_only : negative control; the same image occurs in positive/negative rows
  text_only  : candidate gamma-pair features only
  cross_only : explicit text-image interaction features only
  text_image : image + gamma + interaction features (main probe)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

VERSION = "2026-07-17-matched-linkability-warden-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/evaluate matched linkability wardens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--views",
        nargs="+",
        default=["image_only", "text_only", "cross_only", "text_image"],
        choices=["image_only", "text_only", "cross_only", "text_image"],
    )
    p.add_argument(
        "--classifiers",
        nargs="+",
        default=["logistic", "extra_trees"],
        choices=["logistic", "extra_trees"],
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def feature_columns(df: pd.DataFrame, view: str) -> list[str]:
    image = [c for c in df.columns if c.startswith("img_")]
    gamma = [c for c in df.columns if c.startswith("gamma_")]
    cross = [c for c in df.columns if c.startswith("cross_")]
    mapping = {
        "image_only": image,
        "text_only": gamma,
        "cross_only": cross,
        "text_image": image + gamma + cross,
    }
    cols = mapping[view]
    if not cols:
        raise ValueError(f"No feature columns found for {view}")
    return cols


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


def eer_from_scores(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def calculate_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    pe = 0.5 * (fpr + fnr)
    return {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, score)),
        "eer": eer_from_scores(y, score),
        "pe": float(pe),
        "one_minus_2pe": float(1.0 - 2.0 * pe),
        "abs_advantage": float(abs(1.0 - 2.0 * pe)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def main() -> None:
    args = parse_args()
    print(f"[INFO] Version: {VERSION}")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset does not exist: {args.dataset}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.dataset, low_memory=False)
    required = {"label", "payload_bpp", "cover_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing columns: {sorted(missing)}")

    payloads = sorted(pd.to_numeric(df["payload_bpp"]).round(2).unique())
    records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    jobs = [
        (payload, view, classifier)
        for payload in payloads
        for view in args.views
        for classifier in args.classifiers
    ]
    iterator = tqdm(jobs, desc="warden experiments", unit="job") if tqdm else jobs

    for payload, view, classifier in iterator:
        subset = df[np.isclose(pd.to_numeric(df["payload_bpp"]), payload)].copy()
        cols = feature_columns(subset, view)
        X = subset[cols].replace([np.inf, -np.inf], np.nan)
        y = subset["label"].astype(int).to_numpy()
        groups = subset["cover_id"].astype(str).to_numpy()

        splitter = GroupShuffleSplit(
            n_splits=args.n_splits,
            test_size=args.test_size,
            random_state=args.seed,
        )

        for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model_seed = args.seed + split_id
            model = build_model(classifier, model_seed, args.n_estimators, args.n_jobs)
            model.fit(X.iloc[train_idx], y[train_idx])
            scores = score_model(model, X.iloc[test_idx])
            result = calculate_metrics(y[test_idx], scores)
            result.update({
                "payload_bpp": float(payload),
                "payload_code": f"{int(round(payload*100)):03d}",
                "view": view,
                "classifier": classifier,
                "split_id": split_id,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_covers": int(len(set(groups[train_idx]))),
                "n_test_covers": int(len(set(groups[test_idx]))),
                "n_features": int(len(cols)),
                "seed": model_seed,
            })
            records.append(result)

            if args.save_predictions:
                meta_cols = [
                    "payload_bpp", "payload_code", "cover_id", "stego_id",
                    "true_gamma_pair_id", "candidate_gamma_pair_id", "label",
                ]
                available = [c for c in meta_cols if c in subset.columns]
                pred_df = subset.iloc[test_idx][available].copy()
                pred_df["score"] = scores
                pred_df["prediction"] = (scores >= 0.5).astype(int)
                pred_df["view"] = view
                pred_df["classifier"] = classifier
                pred_df["split_id"] = split_id
                prediction_frames.append(pred_df)

            joblib.dump(
                {
                    "model": model,
                    "feature_columns": cols,
                    "payload_bpp": float(payload),
                    "view": view,
                    "classifier": classifier,
                    "split_id": split_id,
                },
                model_dir / f"{classifier}_{view}_payload_{int(round(payload*100)):03d}_split_{split_id}.joblib",
                compress=3,
            )

    metrics_df = pd.DataFrame(records)
    per_split_path = args.out_dir / "linkability_metrics_per_split.csv"
    metrics_df.to_csv(per_split_path, index=False)

    metric_cols = [
        "acc", "bacc", "auc", "eer", "pe", "one_minus_2pe",
        "abs_advantage", "mcc", "fpr", "fnr",
    ]
    summary = (
        metrics_df
        .groupby(["payload_bpp", "payload_code", "view", "classifier"], as_index=False)
        .agg(
            **{f"{m}_mean": (m, "mean") for m in metric_cols},
            **{f"{m}_std": (m, "std") for m in metric_cols},
            n_splits=("split_id", "count"),
            n_features=("n_features", "first"),
        )
    )
    summary_path = args.out_dir / "linkability_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    pred_path = None
    if prediction_frames:
        pred_path = args.out_dir / "linkability_test_predictions.csv.gz"
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            pred_path, index=False, compression="gzip"
        )

    config_path = args.out_dir / "warden_config.json"
    config_path.write_text(json.dumps({
        "version": VERSION,
        "dataset": str(args.dataset.resolve()),
        "views": args.views,
        "classifiers": args.classifiers,
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
    }, indent=2), encoding="utf-8")

    print("\n[OK] Wrote:")
    for path in [per_split_path, summary_path, pred_path, config_path]:
        if path:
            print("  ", path)


if __name__ == "__main__":
    main()
