#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_gpu_memory_mb():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        vals = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
        return max(vals) if vals else None
    except Exception:
        return None


def load_gray(path: str):
    img = Image.open(path).convert("L")
    return np.asarray(img)


def flatten_feature(feat):
    try:
        import sealwatch as sw
        if isinstance(feat, dict):
            return sw.tools.flatten(feat).astype(np.float32)
    except Exception:
        pass

    if isinstance(feat, dict):
        arrs = []
        for _, v in feat.items():
            arrs.append(np.asarray(v).ravel())
        return np.concatenate(arrs).astype(np.float32)

    return np.asarray(feat).ravel().astype(np.float32)


def extract_srm_feature_with_sealwatch(path: str):
    """
    Uses sealwatch SRM feature extractor.

    The API has changed across versions, so this function tries the common
    extractor forms in a safe order.
    """
    try:
        import sealwatch as sw
    except Exception as e:
        raise RuntimeError(
            "sealwatch is not installed. Run scripts/00_setup_env.sh first."
        ) from e

    if not hasattr(sw, "srm"):
        raise RuntimeError("sealwatch.srm module was not found.")

    # Try file-based extraction first.
    if hasattr(sw.srm, "extract_from_file"):
        try:
            feat = sw.srm.extract_from_file(path)
            return flatten_feature(feat)
        except Exception:
            pass

    if hasattr(sw.srm, "extract"):
        # Some versions accept a path.
        try:
            feat = sw.srm.extract(path)
            return flatten_feature(feat)
        except Exception:
            pass

        # Other versions accept an image array.
        try:
            img = load_gray(path)
            feat = sw.srm.extract(img)
            return flatten_feature(feat)
        except Exception as e:
            raise RuntimeError(f"SRM extraction failed for {path}: {e}") from e

    raise RuntimeError("Could not find a compatible sealwatch SRM extractor.")


def feature_cache_path(image_path: str, cache_dir: Path, feature_kind: str):
    p = Path(image_path).resolve()
    try:
        st = p.stat()
        sig = f"{str(p)}::{st.st_size}::{st.st_mtime_ns}::{feature_kind}"
    except FileNotFoundError:
        sig = f"{str(p)}::missing::{feature_kind}"
    return cache_dir / f"{sha1_text(sig)}.npy"


def get_feature(image_path: str, cache_dir: Path, feature_kind: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = feature_cache_path(image_path, cache_dir, feature_kind)

    if cp.exists():
        return np.load(cp, allow_pickle=False)

    if feature_kind.lower() in {"srm", "maxsrm"}:
        feat = extract_srm_feature_with_sealwatch(image_path)
    else:
        raise ValueError(f"Unsupported feature_kind: {feature_kind}")

    np.save(cp, feat.astype(np.float32))
    return feat.astype(np.float32)


def build_matrix(paths, cache_dir: Path, feature_kind: str):
    X = []
    for p in tqdm(paths, desc=f"Extract/load {feature_kind} features"):
        X.append(get_feature(p, cache_dir, feature_kind))
    return np.vstack(X).astype(np.float32)


@dataclass
class SubspaceModel:
    cols: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    clf: object


class RandomSubspaceFLDEnsemble:
    def __init__(
        self,
        n_estimators=100,
        subspace_dim=2000,
        bootstrap=True,
        seed=20260605,
        verbose=True,
    ):
        self.n_estimators = int(n_estimators)
        self.subspace_dim = int(subspace_dim)
        self.bootstrap = bool(bootstrap)
        self.seed = int(seed)
        self.verbose = verbose
        self.models = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        subspace_dim = min(self.subspace_dim, d)

        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        n_each = min(len(idx0), len(idx1))

        if n_each == 0:
            raise ValueError("Both cover and stego classes are required for training.")

        self.models = []

        iterator = range(self.n_estimators)
        if self.verbose:
            iterator = tqdm(iterator, desc="Train FLD ensemble")

        attempts = 0
        for _ in iterator:
            attempts += 1
            cols = rng.choice(d, size=subspace_dim, replace=False)

            if self.bootstrap:
                b0 = rng.choice(idx0, size=n_each, replace=True)
                b1 = rng.choice(idx1, size=n_each, replace=True)
                train_idx = np.concatenate([b0, b1])
            else:
                train_idx = np.concatenate([idx0, idx1])

            Xs = X[train_idx][:, cols].astype(np.float64)
            ys = y[train_idx]

            mean = Xs.mean(axis=0)
            std = Xs.std(axis=0)
            std[std < 1e-8] = 1.0
            Xs = (Xs - mean) / std

            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

            try:
                clf.fit(Xs, ys)
                self.models.append(SubspaceModel(cols=cols, mean=mean, std=std, clf=clf))
            except Exception:
                continue

        if not self.models:
            raise RuntimeError("No FLD ensemble members could be trained.")

        return self

    def predict_scores(self, X):
        if not self.models:
            raise RuntimeError("Model has not been fitted.")

        scores = np.zeros(X.shape[0], dtype=np.float64)

        for m in self.models:
            Xs = X[:, m.cols].astype(np.float64)
            Xs = (Xs - m.mean) / m.std

            if hasattr(m.clf, "predict_proba"):
                s = m.clf.predict_proba(Xs)[:, 1]
            else:
                dec = m.clf.decision_function(Xs)
                s = 1.0 / (1.0 + np.exp(-dec))

            scores += s

        scores /= len(self.models)
        return scores

    def predict(self, X, threshold=0.5):
        return (self.predict_scores(X) >= threshold).astype(int)


def compute_metrics(y_true, scores, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)

    metrics = {}
    metrics["acc"] = float(accuracy_score(y_true, y_pred))
    metrics["bacc"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    try:
        metrics["auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        metrics["auc"] = float("nan")

    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1.0 - tpr

    idx_eer = int(np.nanargmin(np.abs(fpr - fnr)))
    metrics["eer"] = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)

    pe = np.min((fpr + fnr) / 2.0)
    metrics["pe"] = float(pe)

    for target_fpr in [0.01, 0.05, 0.10]:
        valid = np.where(fpr <= target_fpr)[0]
        if len(valid) == 0:
            metrics[f"tpr_at_fpr_{target_fpr:.2f}"] = 0.0
        else:
            metrics[f"tpr_at_fpr_{target_fpr:.2f}"] = float(np.max(tpr[valid]))

    return metrics


def make_xy(df_split, cache_dir: Path, feature_kind: str):
    cover_paths = df_split["cover_path"].tolist()
    stego_paths = df_split["stego_path"].tolist()

    paths = cover_paths + stego_paths
    labels = np.array([0] * len(cover_paths) + [1] * len(stego_paths), dtype=np.int64)

    X = build_matrix(paths, cache_dir, feature_kind)

    return X, labels, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--feature-kind", default="srm", choices=["srm", "maxsrm"])
    ap.add_argument("--payload-code", default=None, help="Optional payload code check, e.g. 010, 020, 030, 040")
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--subspace-dim", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-splits", default="val,test")
    args = ap.parse_args()

    start = time.time()

    out_dir = Path(args.out_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pairs_csv, dtype={"payload_code": str})

    if args.payload_code is not None:
        expected = str(args.payload_code).strip().zfill(3)

        if "payload_code" not in df.columns:
            raise ValueError("pairs CSV has no payload_code column for checking.")

        got = set(str(x).strip().zfill(3) for x in df["payload_code"].dropna().unique())

        if got != {expected}:
            raise ValueError(
                f"Payload mismatch: command requested payload {expected}, "
                f"but pairs CSV contains payload_code values {sorted(got)}"
            )

        print(f"[INFO] Payload check passed: payload_{expected}")

    required = {"split", "cover_path", "stego_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Pairs CSV missing columns: {missing}")

    config = vars(args).copy()
    config["start_time"] = now_iso()

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    df_train = df[df["split"] == args.train_split].copy()
    if df_train.empty:
        raise ValueError(f"No rows found for train split: {args.train_split}")

    print(f"[INFO] Training pairs: {len(df_train)}")

    X_train, y_train, train_paths = make_xy(df_train, cache_dir, args.feature_kind)

    model = RandomSubspaceFLDEnsemble(
        n_estimators=args.n_estimators,
        subspace_dim=args.subspace_dim,
        bootstrap=True,
        seed=args.seed,
        verbose=True,
    )

    model.fit(X_train, y_train)

    joblib.dump(model, out_dir / "ec_model.joblib")

    all_metrics = []
    all_predictions = []

    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    for split in eval_splits:
        df_eval = df[df["split"] == split].copy()
        if df_eval.empty:
            print(f"[WARN] No rows for eval split: {split}")
            continue

        print(f"[INFO] Evaluating split={split}, pairs={len(df_eval)}")

        X_eval, y_eval, eval_paths = make_xy(df_eval, cache_dir, args.feature_kind)
        scores = model.predict_scores(X_eval)
        preds = (scores >= args.threshold).astype(int)

        met = compute_metrics(y_eval, scores, threshold=args.threshold)
        met["split"] = split
        met["n_samples"] = int(len(y_eval))
        met["n_pairs"] = int(len(df_eval))
        met["feature_kind"] = args.feature_kind
        met["n_estimators_requested"] = int(args.n_estimators)
        met["n_estimators_trained"] = int(len(model.models))
        met["subspace_dim"] = int(args.subspace_dim)
        met["threshold"] = float(args.threshold)
        met["seed"] = int(args.seed)

        all_metrics.append(met)

        pred_df = pd.DataFrame({
            "split": split,
            "path": eval_paths,
            "label": y_eval,
            "score_stego": scores,
            "pred": preds,
        })
        all_predictions.append(pred_df)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            out_dir / "predictions.csv", index=False
        )

    elapsed = time.time() - start

    resource = {
        "finish_time": now_iso(),
        "elapsed_sec": elapsed,
        "feature_kind": args.feature_kind,
        "gpu_memory_used_mb": get_gpu_memory_mb(),
        "pid": os.getpid(),
    }

    try:
        import psutil
        p = psutil.Process(os.getpid())
        resource["rss_memory_mb"] = p.memory_info().rss / (1024 ** 2)
        resource["cpu_percent"] = p.cpu_percent(interval=1.0)
    except Exception:
        resource["rss_memory_mb"] = None
        resource["cpu_percent"] = None

    with open(out_dir / "resource_log.json", "w") as f:
        json.dump(resource, f, indent=2)

    print("\n[OK] Finished.")
    print(metrics_df)
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()