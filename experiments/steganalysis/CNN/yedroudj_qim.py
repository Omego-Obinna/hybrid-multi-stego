#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yedroudj_qim.py

Yedroudj-Net detector runner for Ours-QIM-v3.2-Fused-CF.
Place this file in the same directory as the official PyTorch yed.py,
srm_filter_kernel.py, and MPNCOV.py.
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import psutil
except Exception:
    psutil = None

from yed import Net, initWeights

DEFAULT_DATA_ROOT = Path("/home/omego/Documents/Hybrid_Stego_Eva/yenet_main_split_v3_2_fused075")
DEFAULT_OUT_ROOT = Path("/home/omego/Documents/Hybrid_Stego_Eva/yedroudj_main_results_v3_2_fused075")
IMAGE_EXTS = {".pgm", ".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


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
        return x
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
        torch.backends.cudnn.benchmark = True


def json_safe_args(args) -> Dict[str, object]:
    out = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def get_ram_stats() -> Dict[str, float]:
    if psutil is None:
        return {"process_rss_mb": float("nan"), "ram_total_mb": float("nan"), "ram_used_mb": float("nan"), "ram_available_mb": float("nan")}
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
        return {"gpu_available": False, "gpu_name": "", "gpu_mem_allocated_mb": 0.0, "gpu_mem_reserved_mb": 0.0, "gpu_max_mem_allocated_mb": 0.0, "gpu_max_mem_reserved_mb": 0.0}
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
        p = subprocess.run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return ""


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    sum_pos = float(full_ranks[pos].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def compute_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = np.asarray(preds).astype(int)
    acc = float((preds == labels).mean() * 100.0) if labels.size else float("nan")
    auc = auc_rank(labels, probs)
    cover = labels == 0
    stego = labels == 1
    return {
        "accuracy_pct": acc,
        "auc": auc,
        "error_rate_pct": 100.0 - acc if math.isfinite(acc) else float("nan"),
        "cover_accuracy_pct": float((preds[cover] == 0).mean() * 100.0) if cover.any() else float("nan"),
        "stego_accuracy_pct": float((preds[stego] == 1).mean() * 100.0) if stego.any() else float("nan"),
        "false_alarm_rate_pct": float((preds[cover] == 1).mean() * 100.0) if cover.any() else float("nan"),
        "missed_detection_rate_pct": float((preds[stego] == 0).mean() * 100.0) if stego.any() else float("nan"),
    }


class PairedCoverStegoDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 512, augment: bool = False, allow_resize: bool = False):
        self.root = Path(root)
        self.cover_dir = self.root / "cover"
        self.stego_dir = self.root / "stego"
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.allow_resize = bool(allow_resize)
        if not self.cover_dir.exists():
            raise FileNotFoundError(f"Missing cover dir: {self.cover_dir}")
        if not self.stego_dir.exists():
            raise FileNotFoundError(f"Missing stego dir: {self.stego_dir}")
        cover_names = sorted([p.name for p in self.cover_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        stego_names = sorted([p.name for p in self.stego_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if cover_names != stego_names:
            raise RuntimeError(f"Cover/stego filenames do not match in {self.root}")
        if not cover_names:
            raise RuntimeError(f"No paired images found in {self.root}")
        self.names = cover_names

    def __len__(self) -> int:
        return len(self.names)

    def _read_gray(self, path: Path) -> np.ndarray:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Could not read image: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if not self.allow_resize and arr.shape[:2] != (self.image_size, self.image_size):
            raise RuntimeError(f"Unexpected image shape for {path}: {arr.shape}; expected {self.image_size}x{self.image_size}")
        if self.allow_resize and arr.shape[:2] != (self.image_size, self.image_size):
            arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return arr

    def __getitem__(self, idx: int) -> Dict[str, object]:
        name = self.names[idx]
        cover_path = self.cover_dir / name
        stego_path = self.stego_dir / name
        cover = self._read_gray(cover_path)
        stego = self._read_gray(stego_path)
        if self.augment:
            rot = random.randint(0, 3)
            if rot:
                cover = np.rot90(cover, rot).copy()
                stego = np.rot90(stego, rot).copy()
            if random.random() < 0.5:
                cover = np.flip(cover, axis=1).copy()
                stego = np.flip(stego, axis=1).copy()
        cover = torch.from_numpy(cover.astype(np.float32)).unsqueeze(0)
        stego = torch.from_numpy(stego.astype(np.float32)).unsqueeze(0)
        return {"cover": cover, "stego": stego, "filename": name, "cover_path": str(cover_path), "stego_path": str(stego_path)}


def collate_pairs(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "cover": torch.stack([b["cover"] for b in batch], dim=0),
        "stego": torch.stack([b["stego"] for b in batch], dim=0),
        "filename": [b["filename"] for b in batch],
        "cover_path": [b["cover_path"] for b in batch],
        "stego_path": [b["stego_path"] for b in batch],
    }


def make_loader(root: Path, batch_size: int, shuffle: bool, workers: int, image_size: int, augment: bool, allow_resize: bool) -> DataLoader:
    ds = PairedCoverStegoDataset(root, image_size=image_size, augment=augment, allow_resize=allow_resize)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, collate_fn=collate_pairs)


def batch_to_inputs_labels(batch: Dict[str, object], device: torch.device):
    inputs = torch.cat((batch["cover"], batch["stego"]), dim=0).to(device, dtype=torch.float, non_blocking=True)
    labels = torch.cat((torch.zeros(len(batch["filename"]), dtype=torch.long), torch.ones(len(batch["filename"]), dtype=torch.long)), dim=0).to(device, non_blocking=True)
    filenames = list(batch["filename"]) + list(batch["filename"])
    paths = list(batch["cover_path"]) + list(batch["stego_path"])
    return inputs, labels, filenames, paths


def run_epoch(model, loader, loss_fn, device, optimizer=None, desc: str = "") -> Dict[str, object]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses, all_labels, all_probs, all_preds = [], [], [], []
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
            m = compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs), np.concatenate(all_preds))
            stream.set_postfix(loss=np.mean(losses), acc=m["accuracy_pct"], auc=m["auc"])
    labels_np = np.concatenate(all_labels) if all_labels else np.array([])
    probs_np = np.concatenate(all_probs) if all_probs else np.array([])
    preds_np = np.concatenate(all_preds) if all_preds else np.array([])
    metrics = compute_metrics(labels_np, probs_np, preds_np)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["n_samples"] = int(labels_np.size)
    return metrics


def run_test_with_rows(model, loader, loss_fn, device, payload_label: str):
    model.eval()
    losses, all_labels, all_probs, all_preds, rows = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"payload_{payload_label} test", unit="batch"):
            inputs, labels, filenames, paths = batch_to_inputs_labels(batch, device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            logits = outputs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            losses.append(float(loss.item()))
            all_labels.append(labels_np)
            all_probs.append(probs_np)
            all_preds.append(preds_np)
            for i in range(len(labels_np)):
                rows.append({
                    "payload_label": payload_label, "filename": filenames[i], "image_path": paths[i],
                    "label": int(labels_np[i]), "label_name": "stego" if int(labels_np[i]) else "cover",
                    "prob_stego": float(probs_np[i]), "pred": int(preds_np[i]),
                    "pred_name": "stego" if int(preds_np[i]) else "cover",
                    "correct": int(preds_np[i] == labels_np[i]),
                    "logit_cover": float(logits[i, 0]), "logit_stego": float(logits[i, 1]),
                })
    labels_np = np.concatenate(all_labels) if all_labels else np.array([])
    probs_np = np.concatenate(all_probs) if all_probs else np.array([])
    preds_np = np.concatenate(all_preds) if all_preds else np.array([])
    metrics = compute_metrics(labels_np, probs_np, preds_np)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["n_samples"] = int(labels_np.size)
    return metrics, rows


EPOCH_HEADER = ["timestamp", "payload_label", "payload_bpp", "epoch", "epochs", "optimizer", "lr", "momentum", "weight_decay", "train_loss", "train_accuracy_pct", "train_auc", "val_loss", "val_accuracy_pct", "val_auc", "val_error_rate_pct", "val_false_alarm_rate_pct", "val_missed_detection_rate_pct", "epoch_time_sec", "device", "gpu_name", "process_rss_mb", "ram_total_mb", "ram_used_mb", "ram_available_mb", "gpu_mem_allocated_mb", "gpu_mem_reserved_mb", "gpu_max_mem_allocated_mb", "gpu_max_mem_reserved_mb"]
SUMMARY_HEADER = ["timestamp", "detector", "payload_label", "payload_bpp", "train_pairs", "val_pairs", "test_pairs", "test_samples", "best_epoch", "best_val_auc", "best_val_accuracy_pct", "test_loss", "test_accuracy_pct", "test_auc", "test_error_rate_pct", "test_cover_accuracy_pct", "test_stego_accuracy_pct", "test_false_alarm_rate_pct", "test_missed_detection_rate_pct", "total_time_sec", "train_time_sec", "test_time_sec", "optimizer", "initial_lr", "final_lr", "momentum", "weight_decay", "epochs", "batch_size", "image_size", "augment", "device", "gpu_name", "cuda_available", "torch_version", "hostname", "process_rss_mb", "ram_total_mb", "ram_used_mb", "ram_available_mb", "gpu_mem_allocated_mb", "gpu_mem_reserved_mb", "gpu_max_mem_allocated_mb", "gpu_max_mem_reserved_mb", "nvidia_smi", "checkpoint_path", "data_root", "output_dir"]
TEST_ROW_HEADER = ["payload_label", "filename", "image_path", "label", "label_name", "prob_stego", "pred", "pred_name", "correct", "logit_cover", "logit_stego"]


def run_payload(args, payload_label: str, device: torch.device) -> Dict[str, object]:
    payload_bpp = {"010": 0.10, "020": 0.20, "030": 0.30, "040": 0.40}[payload_label]
    payload_root = args.data_root / f"payload_{payload_label}"
    out_dir = args.out_root / f"payload_{payload_label}"
    ckpt_dir, logs_dir = out_dir / "checkpoints", out_dir / "logs"
    mkdirp(ckpt_dir); mkdirp(logs_dir)
    train_loader = make_loader(payload_root / "train", args.batch_size, True, args.num_workers, args.image_size, args.augment, args.allow_resize)
    val_loader = make_loader(payload_root / "val", args.batch_size, False, args.num_workers, args.image_size, False, args.allow_resize)
    test_loader = make_loader(payload_root / "test", args.batch_size, False, args.num_workers, args.image_size, False, args.allow_resize)
    model = Net().to(device)
    model.apply(initWeights)
    loss_fn = nn.CrossEntropyLoss()
    params_wd, params_rest = [], []
    for p in model.parameters():
        if p.requires_grad:
            (params_wd if p.dim() != 1 else params_rest).append(p)
    if args.optimizer == "sgd":
        optimizer = optim.SGD([{"params": params_wd, "weight_decay": args.weight_decay}, {"params": params_rest, "weight_decay": 0.0}], lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in args.milestones.split(",") if x.strip()], gamma=args.gamma)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    payload_start = time.perf_counter(); train_start = time.perf_counter()
    best_val_auc, best_val_acc, best_epoch = -1.0, -1.0, -1
    best_ckpt = ckpt_dir / "best_yedroudj.pt"
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer, f"payload_{payload_label} epoch {epoch}/{args.epochs} train")
        va = run_epoch(model, val_loader, loss_fn, device, None, f"payload_{payload_label} epoch {epoch}/{args.epochs} val")
        if scheduler is not None: scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])
        improved = (float(va["auc"]) > best_val_auc) or (math.isclose(float(va["auc"]), best_val_auc) and float(va["accuracy_pct"]) > best_val_acc)
        if improved:
            best_val_auc, best_val_acc, best_epoch = float(va["auc"]), float(va["accuracy_pct"]), epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "payload_label": payload_label, "payload_bpp": payload_bpp, "val_auc": best_val_auc, "val_accuracy_pct": best_val_acc, "args": json_safe_args(args)}, best_ckpt)
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "payload_label": payload_label, "payload_bpp": payload_bpp, "val_auc": float(va["auc"]), "val_accuracy_pct": float(va["accuracy_pct"]), "args": json_safe_args(args)}, ckpt_dir / f"checkpoint_{epoch:03d}.pt")
        res = {**get_ram_stats(), **get_gpu_stats(device)}
        row = {"timestamp": now_iso(), "payload_label": payload_label, "payload_bpp": payload_bpp, "epoch": epoch, "epochs": args.epochs, "optimizer": args.optimizer, "lr": lr, "momentum": args.momentum if args.optimizer == "sgd" else "", "weight_decay": args.weight_decay, "train_loss": tr["loss"], "train_accuracy_pct": tr["accuracy_pct"], "train_auc": tr["auc"], "val_loss": va["loss"], "val_accuracy_pct": va["accuracy_pct"], "val_auc": va["auc"], "val_error_rate_pct": va["error_rate_pct"], "val_false_alarm_rate_pct": va["false_alarm_rate_pct"], "val_missed_detection_rate_pct": va["missed_detection_rate_pct"], "epoch_time_sec": time.perf_counter() - epoch_start, "device": str(device), "gpu_name": res.get("gpu_name", ""), **res}
        append_csv(logs_dir / "yedroudj_epoch_log.csv", EPOCH_HEADER, row)
        append_jsonl(logs_dir / "yedroudj_epoch_log.jsonl", row)
    train_time = time.perf_counter() - train_start
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_start = time.perf_counter()
    te, test_rows = run_test_with_rows(model, test_loader, loss_fn, device, payload_label)
    test_time = time.perf_counter() - test_start
    for r in test_rows:
        append_csv(logs_dir / "yedroudj_test_predictions.csv", TEST_ROW_HEADER, r)
    total_time = time.perf_counter() - payload_start
    res = {**get_ram_stats(), **get_gpu_stats(device)}
    summary = {"timestamp": now_iso(), "detector": "Yedroudj-Net", "payload_label": payload_label, "payload_bpp": payload_bpp, "train_pairs": len(train_loader.dataset), "val_pairs": len(val_loader.dataset), "test_pairs": len(test_loader.dataset), "test_samples": te["n_samples"], "best_epoch": best_epoch, "best_val_auc": best_val_auc, "best_val_accuracy_pct": best_val_acc, "test_loss": te["loss"], "test_accuracy_pct": te["accuracy_pct"], "test_auc": te["auc"], "test_error_rate_pct": te["error_rate_pct"], "test_cover_accuracy_pct": te["cover_accuracy_pct"], "test_stego_accuracy_pct": te["stego_accuracy_pct"], "test_false_alarm_rate_pct": te["false_alarm_rate_pct"], "test_missed_detection_rate_pct": te["missed_detection_rate_pct"], "total_time_sec": total_time, "train_time_sec": train_time, "test_time_sec": test_time, "optimizer": args.optimizer, "initial_lr": args.lr, "final_lr": float(optimizer.param_groups[0]["lr"]), "momentum": args.momentum if args.optimizer == "sgd" else "", "weight_decay": args.weight_decay, "epochs": args.epochs, "batch_size": args.batch_size, "image_size": args.image_size, "augment": args.augment, "device": str(device), "gpu_name": res.get("gpu_name", ""), "cuda_available": torch.cuda.is_available(), "torch_version": torch.__version__, "hostname": socket.gethostname(), **res, "nvidia_smi": nvidia_smi_summary(), "checkpoint_path": str(best_ckpt), "data_root": str(payload_root), "output_dir": str(out_dir)}
    append_csv(args.out_root / "yedroudj_summary_all_payloads.csv", SUMMARY_HEADER, summary)
    append_csv(logs_dir / "yedroudj_summary.csv", SUMMARY_HEADER, summary)
    append_jsonl(logs_dir / "yedroudj_summary.jsonl", summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yedroudj-Net detector for Ours-QIM-v3.2-Fused-CF stego images.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--payloads", default="010,020,030,040")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--allow-resize", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["multistep", "step", "none"], default="multistep")
    p.add_argument("--milestones", default="40,60")
    p.add_argument("--step-size", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--require-cuda", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--save-every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible(args.seed, args.deterministic)
    mkdirp(args.out_root)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] detector=Yedroudj-Net device={device} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available(): print(f"[INFO] gpu={torch.cuda.get_device_name(0)} | {nvidia_smi_summary()}")
    valid = {"010", "020", "030", "040"}
    payloads = [x.strip() for x in args.payloads.split(",") if x.strip()]
    for payload in payloads:
        if payload not in valid: raise ValueError(f"Unknown payload {payload}; use 010,020,030,040")
    start = time.perf_counter()
    for payload in payloads:
        print("=" * 80)
        s = run_payload(args, payload, device)
        print(f"[OK] payload_{payload}: test_acc={s['test_accuracy_pct']:.2f}% test_auc={s['test_auc']:.4f}")
    print(f"[DONE] all payloads completed in {time.perf_counter() - start:.2f} sec")
    print(f"[DONE] combined summary: {args.out_root/'yedroudj_summary_all_payloads.csv'}")


if __name__ == "__main__":
    main()
