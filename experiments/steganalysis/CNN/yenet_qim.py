#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yenet_qim_smoke_earlystop.py — PyTorch 2.6 checkpoint fix + optional early stopping

Ye-Net smoke-test trainer/evaluator for Ours-QIM-v3.2 fused S-UNIWARD/MiPOD stego images.

Place this file inside:
  /home/omego/Documents/Hybrid_Stego_Eva/Deep-Steganalysis-main

It uses the split created by create_yenet_smoke_split.py:
  /home/omego/Documents/Hybrid_Stego_Eva/yenet_smoke_split_v3_2_fused075/
    payload_010/{train,val,test}/{cover,stego}
    payload_020/{train,val,test}/{cover,stego}
    payload_040/{train,val,test}/{cover,stego}

It trains Ye-Net from scratch on each payload using SGD + momentum by default,
runs on CUDA, and writes detailed CSV/JSONL logs for accuracy, AUC, timing,
RAM, and GPU memory.

Interpretation:
  Accuracy/AUC close to 50%/0.5 means the detector is not reliably separating
  cover and stego images in this smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm

try:
    import psutil
except Exception:
    psutil = None


# ---------------------------------------------------------------------
# Paths matching your current experiment
# ---------------------------------------------------------------------

DEFAULT_DATA_ROOT = Path("/home/omego/Documents/Hybrid_Stego_Eva/yenet_smoke_split_v3_2_fused075")
DEFAULT_OUT_ROOT = Path("/home/omego/Documents/Hybrid_Stego_Eva/yenet_smoke_results_v3_2_fused075")


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mkdirp(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, fieldnames: List[str], row: Dict) -> None:
    mkdirp(path.parent)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_jsonl(path: Path, row: Dict) -> None:
    mkdirp(path.parent)
    def convert(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, Path):
            return str(x)
        return str(x)
    with path.open("a") as f:
        f.write(json.dumps(row, default=convert, separators=(",", ":")) + "\n")


def set_reproducible(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster for fixed 512x512 image shape.
        torch.backends.cudnn.benchmark = True


def get_ram_stats() -> Dict[str, float]:
    if psutil is None:
        return {
            "process_rss_mb": float("nan"),
            "ram_total_mb": float("nan"),
            "ram_used_mb": float("nan"),
            "ram_available_mb": float("nan"),
        }
    proc = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    return {
        "process_rss_mb": proc.memory_info().rss / (1024 ** 2),
        "ram_total_mb": vm.total / (1024 ** 2),
        "ram_used_mb": vm.used / (1024 ** 2),
        "ram_available_mb": vm.available / (1024 ** 2),
    }


def get_gpu_stats(device: torch.device) -> Dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "gpu_name": "",
            "gpu_mem_allocated_mb": 0.0,
            "gpu_mem_reserved_mb": 0.0,
            "gpu_max_mem_allocated_mb": 0.0,
            "gpu_max_mem_reserved_mb": 0.0,
        }
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(idx),
        "gpu_mem_allocated_mb": torch.cuda.memory_allocated(idx) / (1024 ** 2),
        "gpu_mem_reserved_mb": torch.cuda.memory_reserved(idx) / (1024 ** 2),
        "gpu_max_mem_allocated_mb": torch.cuda.max_memory_allocated(idx) / (1024 ** 2),
        "gpu_max_mem_reserved_mb": torch.cuda.max_memory_reserved(idx) / (1024 ** 2),
    }


def nvidia_smi_summary() -> str:
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return ""


def json_safe_args(args) -> Dict[str, object]:
    """Convert argparse namespace to checkpoint-safe primitive values."""
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    ROC AUC using average ranks, no sklearn dependency.
    labels: 0 for cover, 1 for stego.
    scores: probability/logit score for class stego.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=float)

    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # ranks are 1-based; average rank for ties
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[i:j + 1] = avg_rank
        i = j + 1

    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    sum_pos_ranks = float(full_ranks[pos].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def compute_classification_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = np.asarray(preds).astype(int)

    correct = preds == labels
    acc = float(correct.mean() * 100.0) if labels.size else float("nan")
    auc = auc_rank(labels, probs)

    cover = labels == 0
    stego = labels == 1

    cover_acc = float((preds[cover] == 0).mean() * 100.0) if cover.any() else float("nan")
    stego_acc = float((preds[stego] == 1).mean() * 100.0) if stego.any() else float("nan")

    false_alarm = float((preds[cover] == 1).mean() * 100.0) if cover.any() else float("nan")
    missed_detection = float((preds[stego] == 0).mean() * 100.0) if stego.any() else float("nan")

    return {
        "accuracy_pct": acc,
        "auc": auc,
        "error_rate_pct": 100.0 - acc if math.isfinite(acc) else float("nan"),
        "cover_accuracy_pct": cover_acc,
        "stego_accuracy_pct": stego_acc,
        "false_alarm_rate_pct": false_alarm,
        "missed_detection_rate_pct": missed_detection,
    }


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class PairedCoverStegoDataset(Dataset):
    """
    Paired cover/stego dataset.

    Each item returns one cover image and its matching stego image.
    The training loop concatenates them into a 2B batch with labels:
      cover -> 0, stego -> 1.
    """

    def __init__(self, root: Path, image_size: int = 512, allow_resize: bool = False):
        self.root = Path(root)
        self.cover_dir = self.root / "cover"
        self.stego_dir = self.root / "stego"
        self.image_size = int(image_size)
        self.allow_resize = bool(allow_resize)

        if not self.cover_dir.exists():
            raise FileNotFoundError(f"Missing cover dir: {self.cover_dir}")
        if not self.stego_dir.exists():
            raise FileNotFoundError(f"Missing stego dir: {self.stego_dir}")

        cover_names = sorted([p.name for p in self.cover_dir.iterdir() if p.is_file()])
        stego_names = sorted([p.name for p in self.stego_dir.iterdir() if p.is_file()])

        if cover_names != stego_names:
            cover_only = sorted(set(cover_names) - set(stego_names))[:5]
            stego_only = sorted(set(stego_names) - set(cover_names))[:5]
            raise RuntimeError(
                f"Cover/stego filenames do not match in {self.root}. "
                f"cover_only={cover_only}, stego_only={stego_only}"
            )

        if not cover_names:
            raise RuntimeError(f"No paired images found in {self.root}")

        self.names = cover_names
        transforms = []
        if self.allow_resize:
            transforms.append(T.Resize((self.image_size, self.image_size)))
        transforms.append(T.ToTensor())
        self.transform = T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.names)

    def _load_gray(self, path: Path) -> torch.Tensor:
        im = Image.open(path).convert("L")
        if not self.allow_resize and im.size != (self.image_size, self.image_size):
            raise RuntimeError(
                f"Unexpected image size for {path}: {im.size}; expected "
                f"{self.image_size}x{self.image_size}. Use --allow-resize only for debugging."
            )
        return self.transform(im)

    def __getitem__(self, index: int) -> Dict[str, object]:
        name = self.names[index]
        cover_path = self.cover_dir / name
        stego_path = self.stego_dir / name
        cover = self._load_gray(cover_path)
        stego = self._load_gray(stego_path)
        return {
            "cover": cover,
            "stego": stego,
            "label": [torch.tensor(0, dtype=torch.long), torch.tensor(1, dtype=torch.long)],
            "filename": name,
            "cover_path": str(cover_path),
            "stego_path": str(stego_path),
        }


def collate_pairs(batch: List[Dict[str, object]]) -> Dict[str, object]:
    covers = torch.stack([b["cover"] for b in batch], dim=0)
    stegos = torch.stack([b["stego"] for b in batch], dim=0)
    cover_labels = torch.zeros(len(batch), dtype=torch.long)
    stego_labels = torch.ones(len(batch), dtype=torch.long)
    return {
        "cover": covers,
        "stego": stegos,
        "label": [cover_labels, stego_labels],
        "filename": [b["filename"] for b in batch],
        "cover_path": [b["cover_path"] for b in batch],
        "stego_path": [b["stego_path"] for b in batch],
    }


def make_loader(root: Path, batch_size: int, shuffle: bool, workers: int, image_size: int, allow_resize: bool) -> DataLoader:
    ds = PairedCoverStegoDataset(root=root, image_size=image_size, allow_resize=allow_resize)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_pairs,
    )


# ---------------------------------------------------------------------
# Train/eval loops
# ---------------------------------------------------------------------

def batch_to_inputs_labels(batch: Dict[str, object], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    inputs = torch.cat((batch["cover"], batch["stego"]), dim=0).to(device, dtype=torch.float, non_blocking=True)
    labels = torch.cat((batch["label"][0], batch["label"][1]), dim=0).to(device, dtype=torch.long, non_blocking=True)

    # Filenames appear twice: once for cover, once for stego.
    filenames = list(batch["filename"]) + list(batch["filename"])
    paths = list(batch["cover_path"]) + list(batch["stego_path"])
    return inputs, labels, filenames, paths


def run_epoch(model, loader, loss_fn, device, optimizer=None, desc: str = "") -> Dict[str, object]:
    train_mode = optimizer is not None
    model.train(train_mode)

    losses = []
    all_labels = []
    all_probs = []
    all_preds = []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        stream = tqdm(loader, desc=desc, unit="batch")
        for batch in stream:
            inputs, labels, _, _ = batch_to_inputs_labels(batch, device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            probs = torch.softmax(outputs.detach(), dim=1)[:, 1]
            preds = outputs.detach().argmax(dim=1)

            losses.append(float(loss.item()))
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            m = compute_classification_metrics(
                np.concatenate(all_labels),
                np.concatenate(all_probs),
                np.concatenate(all_preds),
            )
            stream.set_postfix(loss=np.mean(losses), acc=m["accuracy_pct"], auc=m["auc"])

    labels_np = np.concatenate(all_labels) if all_labels else np.array([])
    probs_np = np.concatenate(all_probs) if all_probs else np.array([])
    preds_np = np.concatenate(all_preds) if all_preds else np.array([])
    metrics = compute_classification_metrics(labels_np, probs_np, preds_np)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["n_samples"] = int(labels_np.size)
    return metrics


def run_test_with_rows(model, loader, loss_fn, device, payload_label: str) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model.eval()
    losses = []
    all_labels = []
    all_probs = []
    all_preds = []
    rows = []

    with torch.no_grad():
        stream = tqdm(loader, desc=f"payload_{payload_label} test", unit="batch")
        for batch in stream:
            inputs, labels, filenames, paths = batch_to_inputs_labels(batch, device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            losses.append(float(loss.item()))
            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            logits_np = outputs.cpu().numpy()

            all_labels.append(labels_np)
            all_probs.append(probs_np)
            all_preds.append(preds_np)

            for i in range(len(labels_np)):
                rows.append({
                    "payload_label": payload_label,
                    "filename": filenames[i],
                    "image_path": paths[i],
                    "label": int(labels_np[i]),
                    "label_name": "stego" if int(labels_np[i]) == 1 else "cover",
                    "prob_stego": float(probs_np[i]),
                    "pred": int(preds_np[i]),
                    "pred_name": "stego" if int(preds_np[i]) == 1 else "cover",
                    "correct": int(preds_np[i] == labels_np[i]),
                    "logit_cover": float(logits_np[i, 0]),
                    "logit_stego": float(logits_np[i, 1]),
                })

    labels_np = np.concatenate(all_labels) if all_labels else np.array([])
    probs_np = np.concatenate(all_probs) if all_probs else np.array([])
    preds_np = np.concatenate(all_preds) if all_preds else np.array([])
    metrics = compute_classification_metrics(labels_np, probs_np, preds_np)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["n_samples"] = int(labels_np.size)
    return metrics, rows


# ---------------------------------------------------------------------
# Main payload experiment
# ---------------------------------------------------------------------

EPOCH_HEADER = [
    "timestamp", "payload_label", "payload_bpp", "epoch", "epochs",
    "optimizer", "lr", "momentum", "weight_decay",
    "train_loss", "train_accuracy_pct", "train_auc",
    "val_loss", "val_accuracy_pct", "val_auc", "val_error_rate_pct",
    "val_false_alarm_rate_pct", "val_missed_detection_rate_pct",
    "early_stop_metric", "early_best_metric", "epochs_since_improvement",
    "early_stopped", "stop_reason",
    "epoch_time_sec", "device", "gpu_name",
    "process_rss_mb", "ram_total_mb", "ram_used_mb", "ram_available_mb",
    "gpu_mem_allocated_mb", "gpu_mem_reserved_mb",
    "gpu_max_mem_allocated_mb", "gpu_max_mem_reserved_mb",
]

SUMMARY_HEADER = [
    "timestamp", "payload_label", "payload_bpp",
    "train_pairs", "val_pairs", "test_pairs", "test_samples",
    "best_epoch", "best_val_auc", "best_val_accuracy_pct",
    "test_loss", "test_accuracy_pct", "test_auc", "test_error_rate_pct",
    "test_cover_accuracy_pct", "test_stego_accuracy_pct",
    "test_false_alarm_rate_pct", "test_missed_detection_rate_pct",
    "total_time_sec", "train_time_sec", "test_time_sec",
    "early_stopping", "early_stop_metric", "early_stop_patience", "early_stop_min_delta",
    "min_epochs", "early_stopped", "stop_reason",
    "optimizer", "initial_lr", "final_lr", "momentum", "weight_decay",
    "epochs", "batch_size", "image_size", "device", "gpu_name",
    "cuda_available", "torch_version", "hostname",
    "process_rss_mb", "ram_total_mb", "ram_used_mb", "ram_available_mb",
    "gpu_mem_allocated_mb", "gpu_mem_reserved_mb",
    "gpu_max_mem_allocated_mb", "gpu_max_mem_reserved_mb",
    "nvidia_smi",
    "checkpoint_path", "data_root", "output_dir",
]

TEST_ROW_HEADER = [
    "payload_label", "filename", "image_path", "label", "label_name",
    "prob_stego", "pred", "pred_name", "correct", "logit_cover", "logit_stego",
]


def run_payload(args, payload_label: str, device: torch.device) -> Dict[str, object]:
    payload_bpp = {"010": 0.10, "020": 0.20, "040": 0.40}[payload_label]
    payload_root = args.data_root / f"payload_{payload_label}"

    train_dir = payload_root / "train"
    val_dir = payload_root / "val"
    test_dir = payload_root / "test"

    out_dir = args.out_root / f"payload_{payload_label}"
    ckpt_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    mkdirp(ckpt_dir)
    mkdirp(logs_dir)

    # Configure repository config before importing YeNet.
    import config as c
    c.stego_img_height = int(args.image_size)
    c.stego_img_channel = 1

    from models.YeNet import Model

    train_loader = make_loader(train_dir, args.batch_size, True, args.num_workers, args.image_size, args.allow_resize)
    val_loader = make_loader(val_dir, args.batch_size, False, args.num_workers, args.image_size, args.allow_resize)
    test_loader = make_loader(test_dir, args.batch_size, False, args.num_workers, args.image_size, args.allow_resize)

    model = Model().to(device)
    loss_fn = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if args.scheduler == "multistep":
        milestones = [int(x) for x in args.milestones.split(",") if x.strip()]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    payload_start = time.perf_counter()
    train_start = time.perf_counter()

    best_val_auc = -1.0
    best_val_acc = -1.0
    best_epoch = -1
    best_ckpt = ckpt_dir / "best_yenet.pt"

    early_best_metric = -1.0
    epochs_since_improvement = 0
    early_stopped = False
    stop_reason = ""

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, loss_fn, device, optimizer=optimizer,
            desc=f"payload_{payload_label} epoch {epoch}/{args.epochs} train",
        )
        val_metrics = run_epoch(
            model, val_loader, loss_fn, device, optimizer=None,
            desc=f"payload_{payload_label} epoch {epoch}/{args.epochs} val",
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.perf_counter() - epoch_start

        # Choose the strongest detector checkpoint by validation AUC, then accuracy.
        val_auc = float(val_metrics["auc"])
        val_acc = float(val_metrics["accuracy_pct"])
        improved = (val_auc > best_val_auc) or (math.isclose(val_auc, best_val_auc) and val_acc > best_val_acc)
        if improved:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "payload_label": payload_label,
                "payload_bpp": payload_bpp,
                "val_auc": best_val_auc,
                "val_accuracy_pct": best_val_acc,
                "args": json_safe_args(args),
            }, best_ckpt)

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "payload_label": payload_label,
                "payload_bpp": payload_bpp,
                "val_auc": val_auc,
                "val_accuracy_pct": val_acc,
                "args": json_safe_args(args),
            }, ckpt_dir / f"checkpoint_{epoch:03d}.pt")

        resource_row = {**get_ram_stats(), **get_gpu_stats(device)}
        epoch_row = {
            "timestamp": now_iso(),
            "payload_label": payload_label,
            "payload_bpp": payload_bpp,
            "epoch": epoch,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "lr": current_lr,
            "momentum": args.momentum if args.optimizer == "sgd" else "",
            "weight_decay": args.weight_decay,
            "train_loss": train_metrics["loss"],
            "train_accuracy_pct": train_metrics["accuracy_pct"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy_pct": val_metrics["accuracy_pct"],
            "val_auc": val_metrics["auc"],
            "val_error_rate_pct": val_metrics["error_rate_pct"],
            "val_false_alarm_rate_pct": val_metrics["false_alarm_rate_pct"],
            "val_missed_detection_rate_pct": val_metrics["missed_detection_rate_pct"],
            "early_stop_metric": args.early_stop_metric,
            "early_best_metric": early_best_metric,
            "epochs_since_improvement": epochs_since_improvement,
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
            "epoch_time_sec": epoch_time,
            "device": str(device),
            "gpu_name": resource_row.get("gpu_name", ""),
            **resource_row,
        }
        append_csv(logs_dir / "yenet_epoch_log.csv", EPOCH_HEADER, epoch_row)
        append_jsonl(logs_dir / "yenet_epoch_log.jsonl", epoch_row)

        if args.early_stopping:
            metric_value = val_auc if args.early_stop_metric == "val_auc" else val_acc
            if metric_value > early_best_metric + float(args.early_stop_min_delta):
                early_best_metric = metric_value
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epoch >= int(args.min_epochs) and epochs_since_improvement >= int(args.early_stop_patience):
                early_stopped = True
                stop_reason = (
                    f"No improvement in {args.early_stop_metric} greater than "
                    f"{args.early_stop_min_delta} for {args.early_stop_patience} epochs "
                    f"after minimum {args.min_epochs} epochs."
                )
                print(f"[EARLY STOP] payload_{payload_label}: {stop_reason}")
                break

    train_time = time.perf_counter() - train_start

    # Load best checkpoint and test.
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_start = time.perf_counter()
    test_metrics, test_rows = run_test_with_rows(model, test_loader, loss_fn, device, payload_label)
    test_time = time.perf_counter() - test_start

    for r in test_rows:
        append_csv(logs_dir / "yenet_test_predictions.csv", TEST_ROW_HEADER, r)

    total_time = time.perf_counter() - payload_start
    resource_row = {**get_ram_stats(), **get_gpu_stats(device)}

    summary_row = {
        "timestamp": now_iso(),
        "payload_label": payload_label,
        "payload_bpp": payload_bpp,
        "train_pairs": len(train_loader.dataset),
        "val_pairs": len(val_loader.dataset),
        "test_pairs": len(test_loader.dataset),
        "test_samples": test_metrics["n_samples"],
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_accuracy_pct": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_accuracy_pct": test_metrics["accuracy_pct"],
        "test_auc": test_metrics["auc"],
        "test_error_rate_pct": test_metrics["error_rate_pct"],
        "test_cover_accuracy_pct": test_metrics["cover_accuracy_pct"],
        "test_stego_accuracy_pct": test_metrics["stego_accuracy_pct"],
        "test_false_alarm_rate_pct": test_metrics["false_alarm_rate_pct"],
        "test_missed_detection_rate_pct": test_metrics["missed_detection_rate_pct"],
        "total_time_sec": total_time,
        "train_time_sec": train_time,
        "test_time_sec": test_time,
        "early_stopping": bool(args.early_stopping),
        "early_stop_metric": args.early_stop_metric,
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "min_epochs": int(args.min_epochs),
        "early_stopped": bool(early_stopped),
        "stop_reason": stop_reason,
        "optimizer": args.optimizer,
        "initial_lr": args.lr,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "momentum": args.momentum if args.optimizer == "sgd" else "",
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "device": str(device),
        "gpu_name": resource_row.get("gpu_name", ""),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "hostname": socket.gethostname(),
        **resource_row,
        "nvidia_smi": nvidia_smi_summary(),
        "checkpoint_path": str(best_ckpt),
        "data_root": str(payload_root),
        "output_dir": str(out_dir),
    }

    append_csv(args.out_root / "yenet_smoke_summary_all_payloads.csv", SUMMARY_HEADER, summary_row)
    append_csv(logs_dir / "yenet_summary.csv", SUMMARY_HEADER, summary_row)
    append_jsonl(logs_dir / "yenet_summary.jsonl", summary_row)

    return summary_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ye-Net smoke detector for Ours-QIM-v3.2 fused stego images.")

    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--payloads", default="010,020,040", help="Comma-separated payload labels: 010,020,040")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--allow-resize", action="store_true", help="Debug only. Avoid resizing for steganalysis if possible.")

    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--scheduler", choices=["multistep", "step", "none"], default="multistep")
    p.add_argument("--milestones", default="15,25")
    p.add_argument("--step-size", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.5)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--require-cuda", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--early-stopping", action="store_true",
                   help="Stop training when validation metric does not improve for a fixed patience.")
    p.add_argument("--early-stop-metric", choices=["val_auc", "val_accuracy"], default="val_auc")
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--early-stop-min-delta", type=float, default=0.001)
    p.add_argument("--min-epochs", type=int, default=40)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible(args.seed, deterministic=args.deterministic)
    mkdirp(args.out_root)

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check NVIDIA driver, CUDA-enabled PyTorch installation, and active environment."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] gpu={torch.cuda.get_device_name(0)}")
        print(f"[INFO] nvidia_smi={nvidia_smi_summary()}")

    payloads = [x.strip() for x in args.payloads.split(",") if x.strip()]
    valid = {"010", "020", "040"}
    for p in payloads:
        if p not in valid:
            raise ValueError(f"Unknown payload label {p}. Use one of {sorted(valid)}.")

    all_summaries = []
    global_start = time.perf_counter()
    for payload_label in payloads:
        print("=" * 80)
        print(f"[INFO] Running Ye-Net smoke test for payload_{payload_label}")
        summary = run_payload(args, payload_label, device)
        all_summaries.append(summary)
        print(f"[OK] payload_{payload_label}: test_acc={summary['test_accuracy_pct']:.2f}%, test_auc={summary['test_auc']:.4f}")

    global_time = time.perf_counter() - global_start
    print("=" * 80)
    print(f"[DONE] All payloads completed in {global_time:.2f} sec")
    print(f"[DONE] Combined summary: {args.out_root/'yenet_smoke_summary_all_payloads.csv'}")


if __name__ == "__main__":
    main()
