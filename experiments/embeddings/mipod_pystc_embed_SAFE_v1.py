#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiPOD baseline + pySTC embedding for BOSSBase.

This is a conventional cover-modification (CMO) baseline script. The embedded payload is a direct pseudorandom message
bitstream of length L = round(payload_bpp * H * W), generated from the
provided master secret and image/payload label for reproducibility.

Pipeline per cover image and payload bpp:
  1) L = round(payload_bpp * H * W)
  2) m <- deterministic or random payload bits of length L
  3) Estimate pixel residual variances sigma_i^2 using a MiPOD-style
     denoising/residual variance estimator
  4) Numerically solve MiPOD change rates beta_i under the payload constraint
  5) Convert beta_i to additive costs rho_i = log((1 - 2 beta_i) / beta_i)
  6) stego <- pySTC.hide(m, cover, rho, rho, seed, ...)
  7) verify extraction with pySTC.unhide(...), then log metrics.

The gamma1/gamma2 columns are retained only for CSV-schema compatibility
with the hybrid methods and remain empty unless optional paths are supplied
for traceability. They are not used to generate or mask the embedded payload.

Outputs:
  out-root/
    mipod_0.100/*.pgm
    mipod_0.200/*.pgm
    mipod_0.400/*.pgm
    logs/embed_log_mipod.csv
    logs/embed_log_mipod.jsonl
    logs/embed_summary_mipod.csv
"""


from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import subprocess
import tempfile
import resource
import socket
import sys
import time
import uuid
import getpass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from scipy.signal import convolve2d
except Exception:
    convolve2d = None

try:
    import pywt
except Exception as exc:
    pywt = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import psutil
except Exception:
    psutil = None

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None

# GPU logging is kept schema-compatible, but this CPU/STC baseline does not
# require PyTorch. Avoid importing torch here because CUDA initialisation can
# leave background state that slows or blocks process shutdown on some systems.
torch = None

try:
    import pystc
except Exception:
    pystc = None

# ------------------------------- Basic utilities -------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def read_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.array(im, dtype=np.uint8)


def write_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def mse_psnr_ssim(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    diff = x.astype(np.float64) - y.astype(np.float64)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 0 else float(10.0 * math.log10((255.0 * 255.0) / mse))
    if _ssim is not None:
        try:
            ssim = float(_ssim(x, y, data_range=255))
        except Exception:
            ssim = float("nan")
    else:
        ssim = float("nan")
    return mse, psnr, ssim


def process_rss_mb() -> float:
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        except Exception:
            pass
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def peak_rss_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return process_rss_mb()


def system_ram_info() -> Dict[str, float]:
    if psutil is None:
        return {"ram_total_mb": float("nan"), "ram_available_mb": float("nan"), "ram_used_mb": float("nan")}
    try:
        vm = psutil.virtual_memory()
        return {
            "ram_total_mb": vm.total / (1024 ** 2),
            "ram_available_mb": vm.available / (1024 ** 2),
            "ram_used_mb": vm.used / (1024 ** 2),
        }
    except Exception:
        return {"ram_total_mb": float("nan"), "ram_available_mb": float("nan"), "ram_used_mb": float("nan")}


def cpu_times() -> Dict[str, float]:
    if psutil is None:
        return {"cpu_user_time_sec": float("nan"), "cpu_system_time_sec": float("nan")}
    try:
        t = psutil.Process(os.getpid()).cpu_times()
        return {"cpu_user_time_sec": float(t.user), "cpu_system_time_sec": float(t.system)}
    except Exception:
        return {"cpu_user_time_sec": float("nan"), "cpu_system_time_sec": float("nan")}


def gpu_info() -> Dict[str, object]:
    out = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_mem_allocated_mb": 0.0,
        "gpu_mem_reserved_mb": 0.0,
    }
    if torch is None:
        return out
    try:
        if torch.cuda.is_available():
            out["gpu_available"] = True
            idx = torch.cuda.current_device()
            out["gpu_name"] = torch.cuda.get_device_name(idx)
            out["gpu_mem_allocated_mb"] = float(torch.cuda.memory_allocated(idx) / (1024 ** 2))
            out["gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved(idx) / (1024 ** 2))
    except Exception:
        pass
    return out


def append_csv(path: Path, header: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_jsonl(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=convert, separators=(",", ":")) + "\n")


# ------------------------------- Bit generation -------------------------------

def bytes_to_bits(data: bytes) -> np.ndarray:
    if len(data) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> Tuple[bytes, int]:
    """Pack bits into bytes and return (bytes, number_of_zero_padding_bits)."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-int(bits.size)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes(), int(pad)


def hmac_bytes(key: bytes, msg: bytes, nbytes: int) -> bytes:
    """Expand HMAC-SHA256(key,msg||counter) to nbytes."""
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        out.extend(hmac.new(key, msg + ctr.to_bytes(8, "big"), hashlib.sha256).digest())
        ctr += 1
    return bytes(out[:nbytes])


def hmac_bits(key: bytes, msg: bytes, nbits: int) -> np.ndarray:
    nbytes = (nbits + 7) // 8
    bits = bytes_to_bits(hmac_bytes(key, msg, nbytes))
    return bits[:nbits]


def load_cover_stream_bits(path: Path) -> np.ndarray:
    """Read a text/bytes file and convert to raw UTF-8/byte bits."""
    data = path.read_bytes()
    return bytes_to_bits(data)


def expand_stream_bits(base_bits: np.ndarray, nbits: int, key: bytes, label: str) -> np.ndarray:
    """
    Deterministically expand a cover stream to nbits.
    If the file has enough bits, repeat+truncate with a keyed rotation so different
    images do not always use the same prefix. If not, append HMAC expansion.
    """
    if nbits <= 0:
        return np.zeros(0, dtype=np.uint8)
    if base_bits.size == 0:
        return hmac_bits(key, f"empty-gamma|{label}".encode(), nbits)

    digest = hmac.new(key, f"rotate|{label}".encode(), hashlib.sha256).digest()
    offset = int.from_bytes(digest[:8], "big") % int(base_bits.size)
    rotated = np.concatenate([base_bits[offset:], base_bits[:offset]])
    if rotated.size >= nbits:
        return rotated[:nbits].astype(np.uint8)

    reps = int(math.ceil(nbits / rotated.size))
    tiled = np.tile(rotated, reps)[:nbits]
    # Mix with HMAC expansion to avoid simple periodicity when repeated.
    pad = hmac_bits(key, f"gamma-expand|{label}".encode(), nbits)
    return np.bitwise_xor(tiled.astype(np.uint8), pad.astype(np.uint8))


# ------------------------------- Cost maps -------------------------------

def _resize_to(arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = shape_hw
    H, W = arr.shape
    out = np.zeros((Ht, Wt), dtype=arr.dtype)
    hs, ws = min(H, Ht), min(W, Wt)
    h0, w0 = (H - hs) // 2, (W - ws) // 2
    ho, wo = (Ht - hs) // 2, (Wt - ws) // 2
    out[ho:ho + hs, wo:wo + ws] = arr[h0:h0 + hs, w0:w0 + ws]
    return out


def _uniform_filter2d(x: np.ndarray, size: int) -> np.ndarray:
    """Small wrapper around scipy.ndimage.uniform_filter with a convolution fallback."""
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(x.astype(np.float64), size=int(size), mode="reflect")
    except Exception:
        if convolve2d is None:
            # Very conservative fallback: global variance map.
            return np.full_like(x, float(np.mean(x)), dtype=np.float64)
        k = np.ones((int(size), int(size)), dtype=np.float64) / float(size * size)
        return convolve2d(x.astype(np.float64), k, mode="same", boundary="symm")


def _wiener_denoise(x: np.ndarray, size: int) -> np.ndarray:
    """Wiener content suppression used before MiPOD-style variance estimation.

    SciPy's Wiener filter can emit divide-by-zero/invalid-value warnings on
    uniform or near-uniform local patches. We therefore sanitise its output
    before using it in the residual variance estimate.
    """
    try:
        from scipy.signal import wiener

        out = wiener(x.astype(np.float64), (int(size), int(size))).astype(np.float64)
        out = np.nan_to_num(
            out,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return out

    except Exception:
        out = _uniform_filter2d(x.astype(np.float64), int(size))
        out = np.nan_to_num(
            out,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return out
        
def ternary_entropy_bits(beta: np.ndarray) -> np.ndarray:
    """H(beta) for ±1 changes with probabilities beta,beta and no-change 1-2beta, in bits."""
    beta = np.clip(beta.astype(np.float64), 1e-12, 1.0 / 3.0 - 1e-12)
    p0 = np.clip(1.0 - 2.0 * beta, 1e-12, 1.0)
    return -(2.0 * beta * np.log2(beta) + p0 * np.log2(p0))


def mipod_beta_from_lambda(lambda_val: float, sigma2: np.ndarray, n_iter: int = 32) -> np.ndarray:
    """
    Solve MiPOD Eq. (18) for beta_i in (0,1/3):
        beta_i * sigma_i^{-4} = (1/(2 lambda)) log((1-2 beta_i)/beta_i).
    The vectorised bisection is robust and avoids per-pixel scalar root solving.
    """
    sigma2 = np.maximum(sigma2.astype(np.float64), 1e-8)
    sigma4 = sigma2 * sigma2
    lo = np.full_like(sigma2, 1e-10, dtype=np.float64)
    hi = np.full_like(sigma2, 1.0 / 3.0 - 1e-10, dtype=np.float64)
    lam = max(float(lambda_val), 1e-12)
    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        # f(beta) = log((1-2b)/b) - 2 lambda beta / sigma^4.
        # f decreases from +inf to negative; root where f=0.
        f = np.log((1.0 - 2.0 * mid) / mid) - (2.0 * lam * mid / sigma4)
        lo = np.where(f > 0, mid, lo)
        hi = np.where(f <= 0, mid, hi)
    return np.clip(0.5 * (lo + hi), 1e-10, 1.0 / 3.0 - 1e-10)


def mipod_solve_beta(sigma2: np.ndarray, payload_bits: int, max_outer: int = 42) -> Tuple[np.ndarray, float, float]:
    """Find lambda such that sum_i H(beta_i) ~= payload_bits."""
    target = float(payload_bits)
    N = int(sigma2.size)
    max_payload = float(N) * math.log2(3.0)
    if target <= 0:
        return np.zeros_like(sigma2, dtype=np.float64), float("inf"), 0.0
    if target >= 0.98 * max_payload:
        # Extremely high rate for ternary embedding; cap beta below 1/3.
        beta = np.full_like(sigma2, 0.32, dtype=np.float64)
        return beta, 0.0, float(np.sum(ternary_entropy_bits(beta)))

    lo, hi = 1e-9, 1.0
    # Increase hi until payload is below target.
    for _ in range(80):
        beta_hi = mipod_beta_from_lambda(hi, sigma2, n_iter=24)
        H_hi = float(np.sum(ternary_entropy_bits(beta_hi)))
        if H_hi <= target:
            break
        hi *= 2.0
    best_beta = beta_hi
    best_H = H_hi
    for _ in range(int(max_outer)):
        mid = 0.5 * (lo + hi)
        beta_mid = mipod_beta_from_lambda(mid, sigma2, n_iter=28)
        H_mid = float(np.sum(ternary_entropy_bits(beta_mid)))
        best_beta, best_H = beta_mid, H_mid
        if H_mid > target:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    beta = mipod_beta_from_lambda(lam, sigma2, n_iter=32)
    H = float(np.sum(ternary_entropy_bits(beta)))
    return beta, float(lam), H


def mipod_variance_estimate(
    img_u8: np.ndarray,
    wiener_size: int = 3,
    variance_window: int = 9,
    sigma_floor: float = 1e-4,
) -> np.ndarray:
    """
    MiPOD-style residual variance estimator.
    The original MiPOD uses content suppression followed by a local parametric
    variance estimate. This implementation keeps the same modelling principle:
    suppress local content, estimate residual energy in a local window, and floor
    the variance for numerical stability.
    """
    z = img_u8.astype(np.float64)

    den = _wiener_denoise(z, int(wiener_size))
    den = np.nan_to_num(
        den,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    residual = z - den
    residual = np.nan_to_num(
        residual,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mean_r = _uniform_filter2d(residual, int(variance_window))
    mean_r = np.nan_to_num(
        mean_r,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mean_r2 = _uniform_filter2d(residual * residual, int(variance_window))
    mean_r2 = np.nan_to_num(
        mean_r2,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sigma2 = mean_r2 - mean_r * mean_r
    sigma2 = np.nan_to_num(
        sigma2,
        nan=float(sigma_floor),
        posinf=float(sigma_floor),
        neginf=float(sigma_floor),
    )

    # Numerical floor: prevents zero/negative variance values from creating
    # infinite or degenerate MiPOD costs.
    sigma2 = np.maximum(sigma2, float(sigma_floor))

    # Robust upper cap prevents isolated residuals from making costs degenerate.
    finite = sigma2[np.isfinite(sigma2)]
    if finite.size:
        hi = np.percentile(finite, 99.9)
        sigma2 = np.clip(
            sigma2,
            float(sigma_floor),
            max(float(hi), float(sigma_floor)),
        )
    else:
        sigma2 = np.full_like(z, float(sigma_floor), dtype=np.float64)

    return sigma2.astype(np.float64)
def mipod_cost_map(
    img_u8: np.ndarray,
    payload_bits: int,
    wet_cost: float = 1e10,
    wiener_size: int = 3,
    variance_window: int = 9,
    sigma_floor: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Compute a MiPOD additive cost map:
      1) estimate residual variances sigma_i^2;
      2) solve MiPOD change rates beta_i by minimizing optimal-detector deflection
         under a ternary entropy payload constraint;
      3) convert beta_i to costs rho_i = log((1-2 beta_i)/beta_i);
      4) apply wet costs at saturated pixels.
    """
    sigma2 = mipod_variance_estimate(img_u8, wiener_size, variance_window, sigma_floor)
    beta, lam, achieved_bits = mipod_solve_beta(sigma2, int(payload_bits))
    rho = np.log((1.0 - 2.0 * beta) / beta)
    rho = np.nan_to_num(rho, nan=wet_cost, posinf=wet_cost, neginf=wet_cost)
    rho = np.minimum(rho, wet_cost)
    rho = np.maximum(rho, 1e-6)

    # Boundary-aware symmetric map for pySTC +1/-1 costs.
    rho_p1 = rho.copy()
    rho_m1 = rho.copy()
    rho_p1[img_u8 >= 255] = wet_cost
    rho_m1[img_u8 <= 0] = wet_cost
    rho_sym = np.maximum(rho_p1, rho_m1)

    finite = rho_sym[np.isfinite(rho_sym) & (rho_sym < wet_cost)]
    if finite.size:
        lo, hi = np.percentile(finite, [0.1, 99.9])
        rho_sym = np.clip(rho_sym, max(float(lo), 1e-6), min(float(hi), wet_cost))

    info = {
        "variance_estimator": "wiener_residual_local_variance",
        "wiener_size": int(wiener_size),
        "variance_window": int(variance_window),
        "sigma_floor": float(sigma_floor),
        "mipod_lambda": float(lam),
        "mipod_achieved_payload_bits_entropy": float(achieved_bits),
        "mipod_mean_beta": float(np.mean(beta)),
        "mipod_median_beta": float(np.median(beta)),
        "mipod_max_beta": float(np.max(beta)),
        "mipod_mean_sigma2": float(np.mean(sigma2)),
        "mipod_median_sigma2": float(np.median(sigma2)),
    }
    return rho_sym.astype(np.float32), info


def local_variance_map(img_u8: np.ndarray) -> np.ndarray:
    """3x3 local variance without scipy; sufficient for logging high-variance hit rate."""
    x = img_u8.astype(np.float32)
    pads = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    vals = []
    for dy in range(3):
        for dx in range(3):
            vals.append(pads[dy:dy + x.shape[0], dx:dx + x.shape[1]])
    stack = np.stack(vals, axis=0)
    return np.var(stack, axis=0).astype(np.float32)


# ------------------------------- Embedding -------------------------------

def keyed_noise(shape: Tuple[int, int], key: bytes, label: str) -> np.ndarray:
    n = shape[0] * shape[1]
    raw = hmac_bytes(key, f"noise|{label}".encode(), n * 8)
    vals = np.frombuffer(raw, dtype=">u8").astype(np.float64)
    vals = vals / float(2 ** 64 - 1)
    return vals.reshape(shape)


def embed_cost_adaptive_lsb_matching(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    rho: np.ndarray,
    key: bytes,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Adaptive-cost LSB matching proxy for STC-based embedding.
    Selects L low-cost pixels under a keyed cost tie-breaker, writes payload bits
    to LSBs, and uses ±1 matching with edge guards.

    Returns stego, selected_mask, modified_mask, stats.
    """
    H, W = cover.shape
    N = H * W
    L = int(payload_bits.size)
    if L > N:
        raise ValueError(f"payload length L={L} exceeds image capacity N={N}; reduce bpp or add coding layer")

    rho_flat = rho.reshape(-1).astype(np.float64)
    # Keyed tie-breaker to avoid always selecting the identical set for equal costs.
    noise = keyed_noise((H, W), key, label).reshape(-1)
    score = rho_flat + 1e-9 * noise
    selected_idx = np.argpartition(score, L - 1)[:L] if L > 0 else np.array([], dtype=np.int64)
    # Stable order for extraction.
    selected_idx = selected_idx[np.argsort(score[selected_idx])]

    stego = cover.copy().reshape(-1)
    cover_flat = cover.reshape(-1)
    current_lsb = (stego[selected_idx] & 1).astype(np.uint8)
    need_change = current_lsb != payload_bits
    change_idx = selected_idx[need_change]

    # ±1 LSB matching with deterministic keyed sign.
    sign_bits = hmac_bits(key, f"sign|{label}".encode(), max(1, len(change_idx)))[:len(change_idx)]
    for j, idx in enumerate(change_idx):
        val = int(stego[idx])
        if val <= 0:
            stego[idx] = 1
        elif val >= 255:
            stego[idx] = 254
        else:
            stego[idx] = val + (1 if sign_bits[j] == 1 else -1)

    stego2 = stego.reshape(H, W).astype(np.uint8)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected_idx] = True
    selected_mask = selected_mask.reshape(H, W)
    modified_mask = (cover != stego2)

    flips = int(np.count_nonzero(modified_mask))
    change_rate = flips / float(N)
    payload_bpp_actual = L / float(N)
    embedding_eff = (L / flips) if flips > 0 else float("inf")

    actual_distortion = float(np.sum(rho[modified_mask])) if flips > 0 else 0.0
    if flips > 0:
        lower = float(np.sum(np.partition(rho_flat, flips - 1)[:flips]))
        coding_loss = (actual_distortion / lower - 1.0) if lower > 0 else float("nan")
    else:
        lower = 0.0
        coding_loss = 0.0

    stats = {
        "payload_bits": L,
        "capacity_pixels": N,
        "num_selected_pixels": int(selected_mask.sum()),
        "num_modified_pixels": flips,
        "change_rate": change_rate,
        "actual_payload_bpp": payload_bpp_actual,
        "embedding_efficiency": embedding_eff,
        "actual_distortion": actual_distortion,
        "oracle_min_distortion_same_flips": lower,
        "coding_loss": coding_loss,
        "stc_used": False,
        "stc_mode": "cost_adaptive_lsb_matching_proxy",
    }
    return stego2, selected_mask, modified_mask, stats


def extract_lsb(stego: np.ndarray, selected_mask: np.ndarray, rho: np.ndarray, key: bytes, label: str, L: int) -> np.ndarray:
    """Extraction mirrors sorted selected positions; selected_mask is logged in memory only."""
    # Recompute selected order from rho+key to avoid saving mask.
    H, W = stego.shape
    rho_flat = rho.reshape(-1).astype(np.float64)
    noise = keyed_noise((H, W), key, label).reshape(-1)
    score = rho_flat + 1e-9 * noise
    idx = np.argpartition(score, L - 1)[:L] if L > 0 else np.array([], dtype=np.int64)
    idx = idx[np.argsort(score[idx])]
    return (stego.reshape(-1)[idx] & 1).astype(np.uint8)


def pystc_seed_from_key(key: bytes, label: str) -> int:
    """Derive a deterministic 31-bit seed accepted by pySTC."""
    digest = hmac.new(key, f"pystc-seed|{label}".encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def run_pystc_direct(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    rho: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Run pySTC directly in the current Python process. This is fast but unsafe if the C++ layer aborts."""
    if pystc is None:
        raise RuntimeError("pySTC is not available; install with: pip install pystcstego")
    msg_bytes, padded = bits_to_bytes(payload_bits)
    t0 = time.perf_counter()
    stego = pystc.hide(msg_bytes, cover.astype(np.uint8), rho.astype(np.float64), rho.astype(np.float64), int(seed), mx=255, mn=0)
    t1 = time.perf_counter()
    recovered = pystc.unhide(stego.astype(np.uint8), int(seed))
    t2 = time.perf_counter()
    extracted_bits = bytes_to_bits(bytes(recovered))[:payload_bits.size]
    if extracted_bits.size < payload_bits.size:
        extracted_bits = np.pad(extracted_bits, (0, payload_bits.size - extracted_bits.size), constant_values=0)
    stats = {
        "stc_used": True,
        "stc_mode": "pystc_direct",
        "stc_backend_actual": "pystc",
        "stc_seed": int(seed),
        "stc_message_bytes": len(msg_bytes),
        "stc_padded_bits": int(padded),
        "stc_safe_subprocess": False,
        "worker_returncode": 0,
        "worker_stderr": "",
        "embed_time_sec_backend": float(t1 - t0),
        "extract_time_sec_backend": float(t2 - t1),
    }
    return stego.astype(np.uint8), extracted_bits.astype(np.uint8), stats


def run_pystc_subprocess(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    rho: np.ndarray,
    seed: int,
    timeout_sec: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Run pySTC in a child Python process so C++ aborts do not kill the main experiment."""
    msg_bytes, padded = bits_to_bytes(payload_bits)
    with tempfile.TemporaryDirectory(prefix="mipod_pystc_") as td:
        td_path = Path(td)
        inp = td_path / "worker_in.npz"
        outp = td_path / "worker_out.npz"
        np.savez_compressed(
            inp,
            cover=cover.astype(np.uint8),
            rho=rho.astype(np.float64),
            msg_bytes=np.frombuffer(msg_bytes, dtype=np.uint8),
            seed=np.array([int(seed)], dtype=np.int64),
            L=np.array([int(payload_bits.size)], dtype=np.int64),
        )
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_pystc-worker", "--worker-in", str(inp), "--worker-out", str(outp)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
        stderr_tail = (proc.stderr or "")[-2000:]
        if proc.returncode != 0 or not outp.exists():
            raise RuntimeError(f"pySTC worker failed rc={proc.returncode}: {stderr_tail}")
        data = np.load(outp, allow_pickle=False)
        stego = data["stego"].astype(np.uint8)
        extracted_bits = data["extracted_bits"].astype(np.uint8)[:payload_bits.size]
        if extracted_bits.size < payload_bits.size:
            extracted_bits = np.pad(extracted_bits, (0, payload_bits.size - extracted_bits.size), constant_values=0)
        stats = {
            "stc_used": True,
            "stc_mode": "pystc_safe_subprocess",
            "stc_backend_actual": "pystc",
            "stc_seed": int(seed),
            "stc_message_bytes": len(msg_bytes),
            "stc_padded_bits": int(padded),
            "stc_safe_subprocess": True,
            "worker_returncode": int(proc.returncode),
            "worker_stderr": stderr_tail,
            "embed_time_sec_backend": float(data["embed_time_sec"][0]),
            "extract_time_sec_backend": float(data["extract_time_sec"][0]),
        }
        return stego, extracted_bits, stats


def pystc_worker_main(worker_in: str, worker_out: str) -> int:
    """Hidden worker entry point used by --stc-safe-subprocess."""
    import pystc as _pystc
    data = np.load(worker_in, allow_pickle=False)
    cover = data["cover"].astype(np.uint8)
    rho = data["rho"].astype(np.float64)
    msg = data["msg_bytes"].astype(np.uint8).tobytes()
    seed = int(data["seed"][0])
    L = int(data["L"][0])
    t0 = time.perf_counter()
    stego = _pystc.hide(msg, cover, rho, rho, seed, mx=255, mn=0)
    t1 = time.perf_counter()
    recovered = _pystc.unhide(stego.astype(np.uint8), seed)
    t2 = time.perf_counter()
    extracted_bits = bytes_to_bits(bytes(recovered))[:L]
    np.savez_compressed(
        worker_out,
        stego=stego.astype(np.uint8),
        extracted_bits=extracted_bits.astype(np.uint8),
        embed_time_sec=np.array([float(t1 - t0)], dtype=np.float64),
        extract_time_sec=np.array([float(t2 - t1)], dtype=np.float64),
    )
    return 0


def embed_with_backend(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    rho: np.ndarray,
    key: bytes,
    label: str,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Dispatch embedding to pySTC or the adaptive-cost proxy."""
    requested = args.stc_backend
    seed = pystc_seed_from_key(key, label)
    pystc_available = pystc is not None

    def proxy(mode_suffix: str = "cost_adaptive_lsb_matching_proxy"):
        stego, selected_mask, modified_mask, estats = embed_cost_adaptive_lsb_matching(cover, payload_bits, rho, key, label)
        extracted = extract_lsb(stego, selected_mask, rho, key, label, int(payload_bits.size))
        estats.update({
            "stc_backend_requested": requested,
            "stc_backend_actual": "proxy",
            "pystc_available": bool(pystc_available),
            "stc_seed": int(seed),
            "stc_message_bytes": int((payload_bits.size + 7) // 8),
            "stc_padded_bits": int((-payload_bits.size) % 8),
            "stc_safe_subprocess": bool(args.stc_safe_subprocess),
            "worker_returncode": "",
            "worker_stderr": "",
            "stc_mode": mode_suffix,
        })
        return stego, extracted, selected_mask, modified_mask, estats

    if requested == "proxy":
        return proxy()

    if requested in {"pystc", "auto"}:
        if not pystc_available:
            if requested == "auto":
                return proxy("proxy_pystc_unavailable")
            raise RuntimeError("--stc-backend pystc requested, but pySTC is unavailable. Install with: pip install pystcstego")
        try:
            if args.stc_safe_subprocess:
                stego, extracted, stc_stats = run_pystc_subprocess(cover, payload_bits, rho, seed, args.stc_timeout_sec)
            else:
                stego, extracted, stc_stats = run_pystc_direct(cover, payload_bits, rho, seed)
        except Exception as exc:
            if requested == "auto":
                stego, extracted, selected_mask, modified_mask, estats = proxy("proxy_after_pystc_failure")
                estats["error"] = f"pySTC failed; used proxy fallback: {type(exc).__name__}: {exc}"
                return stego, extracted, selected_mask, modified_mask, estats
            raise
        modified_mask = (cover != stego)
        selected_mask = modified_mask.copy()
        flips = int(np.count_nonzero(modified_mask))
        N = int(cover.size)
        L = int(payload_bits.size)
        actual_distortion = float(np.sum(rho[modified_mask])) if flips else 0.0
        rho_flat = rho.reshape(-1).astype(np.float64)
        if flips > 0:
            lower = float(np.sum(np.partition(rho_flat, flips - 1)[:flips]))
            coding_loss = (actual_distortion / lower - 1.0) if lower > 0 else float("nan")
        else:
            lower = 0.0
            coding_loss = 0.0
        estats = {
            "payload_bits": L,
            "capacity_pixels": N,
            "num_selected_pixels": L,
            "num_modified_pixels": flips,
            "change_rate": flips / float(N),
            "actual_payload_bpp": L / float(N),
            "embedding_efficiency": (L / flips) if flips > 0 else float("inf"),
            "actual_distortion": actual_distortion,
            "oracle_min_distortion_same_flips": lower,
            "coding_loss": coding_loss,
            "stc_backend_requested": requested,
            "pystc_available": bool(pystc_available),
            **stc_stats,
        }
        return stego, extracted, selected_mask, modified_mask, estats

    raise ValueError(f"Unknown stc backend: {requested}")


# ------------------------------- Dataset discovery -------------------------------

def discover_images(root: Path, max_images: Optional[int] = None) -> List[Path]:
    exts = {".pgm", ".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
    paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    paths.sort(key=lambda p: str(p))
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    return paths


def iter_progress(items: List[Tuple[Path, float]], progress: str):
    if tqdm is not None and progress in {"auto", "bar"}:
        yield from tqdm(items, desc="MiPOD embedding", unit="job")
    else:
        total = len(items)
        step = max(1, total // 20)
        for i, item in enumerate(items, 1):
            if progress != "none" and (i == 1 or i % step == 0 or i == total):
                print(f"[MiPOD embedding] {100*i/total:5.1f}% ({i}/{total})", flush=True)
            yield item


# ------------------------------- Logging schema -------------------------------

CSV_HEADER = [
    "run_id", "timestamp", "image_id", "method", "payload_bpp", "payload_length_L",
    "cover_path", "stego_path", "cover_sha256", "stego_sha256",
    "height", "width", "num_pixels", "num_selected_pixels", "num_modified_pixels", "change_rate",
    "actual_payload_bpp", "embedding_efficiency", "coding_loss", "stc_used", "stc_mode",
    "stc_backend_requested", "stc_backend_actual", "pystc_available", "stc_seed",
    "stc_message_bytes", "stc_padded_bits", "stc_safe_subprocess", "worker_returncode", "worker_stderr",
    "mse", "psnr_db", "ssim",
    "mean_cost", "median_cost", "sum_cost", "mean_cost_modified", "modified_in_high_variance_pct",
    "embed_time_sec", "extract_time_sec", "success_extract", "ber",
    "cpu_user_time_sec", "cpu_system_time_sec", "process_rss_mb", "peak_rss_mb",
    "ram_total_mb", "ram_used_mb", "ram_available_mb",
    "gpu_available", "gpu_name", "gpu_mem_allocated_mb", "gpu_mem_reserved_mb",
    "gamma1_path", "gamma2_path", "gamma1_sha256", "gamma2_sha256",
    "seed", "algo_params", "ok", "error"
]

SUMMARY_HEADER = [
    "run_id", "method", "payload_bpp", "n", "fail_rate", "mean_embed_time_sec", "mean_extract_time_sec",
    "mean_ber", "success_rate", "mean_change_rate", "mean_embedding_efficiency", "mean_coding_loss",
    "mean_mse", "median_psnr_db", "mean_ssim", "out_dir"
]


# ------------------------------- Main processing -------------------------------

def process_one(
    cover_path: Path,
    payload_bpp: float,
    args,
    run_id: str,
    gamma1_bits_base: np.ndarray,  # unused; retained for backward-compatible call signature
    gamma2_bits_base: np.ndarray,  # unused; retained for backward-compatible call signature
    master_key: bytes,
    logs_dir: Path,
) -> Dict[str, object]:
    row: Dict[str, object] = {}
    image_id = cover_path.stem
    method = "MiPOD"
    out_dir = Path(args.out_root) / f"mipod_{payload_bpp:.3f}"
    stego_path = out_dir / cover_path.name

    t_embed0 = time.perf_counter()
    cpu0 = cpu_times()
    err = ""
    ok = 1

    try:
        cover = read_gray(cover_path)
        H, W = cover.shape
        N = int(H * W)
        L = int(round(payload_bpp * N))
        if L <= 0:
            raise ValueError("payload length is zero; increase payload_bpp")

        # Per-image/payload keys/labels. For a CMO baseline, the payload is
        # embedded directly; no hybrid gamma masking is applied.
        label = f"{image_id}|{payload_bpp:.3f}|{args.seed}" 
        payload_key = hmac.new(master_key, f"payload|{label}".encode(), hashlib.sha256).digest()

        b_bits = hmac_bits(payload_key, b"mipod-payload", L) if args.deterministic_payload else np.random.default_rng(args.seed).integers(0, 2, L, dtype=np.uint8)

        rho, mipod_info = mipod_cost_map(
            cover,
            payload_bits=L,
            wet_cost=args.wet_cost,
            wiener_size=args.wiener_size,
            variance_window=args.variance_window,
            sigma_floor=args.sigma_floor,
        )
        var_map = local_variance_map(cover)
        high_var_threshold = float(np.percentile(var_map, args.high_variance_percentile))
        high_var_mask = var_map >= high_var_threshold

        embed_key = hmac.new(master_key, f"embed|{label}".encode(), hashlib.sha256).digest()
        t_backend0 = time.perf_counter()
        stego, extracted, selected_mask, modified_mask, estats = embed_with_backend(
            cover, b_bits, rho, embed_key, label, args
        )
        t_backend1 = time.perf_counter()
        write_gray(stego_path, stego)
        t_embed1 = time.perf_counter()

        # Backend routines perform extraction as part of verification. For pySTC safe
        # subprocesses, embed/extract timing comes from the worker; otherwise use the
        # measured total backend time.
        extract_time_backend = float(estats.get("extract_time_sec_backend", 0.0) or 0.0)
        if extract_time_backend <= 0.0:
            extract_time_backend = max(0.0, (t_backend1 - t_backend0) - float(estats.get("embed_time_sec_backend", 0.0) or 0.0))
        t_extract0 = time.perf_counter()
        t_extract1 = t_extract0 + extract_time_backend
        ber = float(np.mean(extracted != b_bits)) if L > 0 and extracted.size == b_bits.size else 1.0
        success_extract = int(ber == 0.0)

        mse, psnr, ssim_val = mse_psnr_ssim(cover, stego)
        cost_mod = rho[modified_mask]
        mean_cost_modified = float(np.mean(cost_mod)) if cost_mod.size else 0.0
        mod_high = float(np.mean(high_var_mask[modified_mask]) * 100.0) if np.count_nonzero(modified_mask) else 0.0

        cpu1 = cpu_times()
        ram = system_ram_info()
        gpu = gpu_info()

        row.update({
            "run_id": run_id,
            "timestamp": now_iso(),
            "image_id": image_id,
            "method": method,
            "payload_bpp": payload_bpp,
            "payload_length_L": L,
            "cover_path": str(cover_path),
            "stego_path": str(stego_path),
            "cover_sha256": sha256_file(cover_path),
            "stego_sha256": sha256_file(stego_path),
            "height": H,
            "width": W,
            "num_pixels": N,
            "num_selected_pixels": estats["num_selected_pixels"],
            "num_modified_pixels": estats["num_modified_pixels"],
            "change_rate": estats["change_rate"],
            "actual_payload_bpp": estats["actual_payload_bpp"],
            "embedding_efficiency": estats["embedding_efficiency"],
            "coding_loss": estats["coding_loss"],
            "stc_used": estats["stc_used"],
            "stc_mode": estats["stc_mode"],
            "stc_backend_requested": estats.get("stc_backend_requested", args.stc_backend),
            "stc_backend_actual": estats.get("stc_backend_actual", "proxy"),
            "pystc_available": estats.get("pystc_available", pystc is not None),
            "stc_seed": estats.get("stc_seed", ""),
            "stc_message_bytes": estats.get("stc_message_bytes", ""),
            "stc_padded_bits": estats.get("stc_padded_bits", ""),
            "stc_safe_subprocess": estats.get("stc_safe_subprocess", False),
            "worker_returncode": estats.get("worker_returncode", ""),
            "worker_stderr": estats.get("worker_stderr", ""),
            "mse": mse,
            "psnr_db": psnr,
            "ssim": ssim_val,
            "mean_cost": float(np.mean(rho)),
            "median_cost": float(np.median(rho)),
            "sum_cost": float(np.sum(rho)),
            "mean_cost_modified": mean_cost_modified,
            "modified_in_high_variance_pct": mod_high,
            "embed_time_sec": t_embed1 - t_embed0,
            "extract_time_sec": t_extract1 - t_extract0,
            "success_extract": success_extract,
            "ber": ber,
            "cpu_user_time_sec": cpu1["cpu_user_time_sec"],
            "cpu_system_time_sec": cpu1["cpu_system_time_sec"],
            "process_rss_mb": process_rss_mb(),
            "peak_rss_mb": peak_rss_mb(),
            **ram,
            **gpu,
            "gamma1_path": str(args.gamma1) if args.gamma1 else "",
            "gamma2_path": str(args.gamma2) if args.gamma2 else "",
            "gamma1_sha256": sha256_file(Path(args.gamma1)) if args.gamma1 and Path(args.gamma1).exists() else "",
            "gamma2_sha256": sha256_file(Path(args.gamma2)) if args.gamma2 and Path(args.gamma2).exists() else "",
            "seed": args.seed,
            "algo_params": json.dumps({
                "baseline": "MiPOD",
                "mask": "none; direct CMO payload",
                "cost_map": "MiPOD variance-estimation + optimal-detector change-rate optimization converted to additive STC costs",
                "wet_cost": args.wet_cost,
                "wiener_size": args.wiener_size,
                "variance_window": args.variance_window,
                "sigma_floor": args.sigma_floor,
                "mipod_info": mipod_info,
                "high_variance_percentile": args.high_variance_percentile,
                "payload_source": "hmac_deterministic" if args.deterministic_payload else "numpy_rng",
                "stc_backend": args.stc_backend,
                "stc_safe_subprocess": args.stc_safe_subprocess,
            }, separators=(",", ":")),
            "ok": ok,
            "error": err,
        })

    except Exception as exc:
        ok = 0
        err = f"{type(exc).__name__}: {exc}"
        cpu1 = cpu_times()
        ram = system_ram_info()
        gpu = gpu_info()
        # Preserve all metadata that was available before the backend failed.
        H_fail = locals().get("H", "")
        W_fail = locals().get("W", "")
        N_fail = locals().get("N", "")
        L_fail = locals().get("L", "")
        rho_fail = locals().get("rho", None)
        embed_key_fail = locals().get("embed_key", b"")
        label_fail = locals().get("label", f"{image_id}|{payload_bpp:.3f}|{getattr(args, 'seed', '')}")
        try:
            stc_seed_fail = pystc_seed_from_key(embed_key_fail, label_fail) if embed_key_fail else ""
        except Exception:
            stc_seed_fail = ""
        mean_cost_fail = float(np.mean(rho_fail)) if isinstance(rho_fail, np.ndarray) else ""
        median_cost_fail = float(np.median(rho_fail)) if isinstance(rho_fail, np.ndarray) else ""
        sum_cost_fail = float(np.sum(rho_fail)) if isinstance(rho_fail, np.ndarray) else ""
        actual_payload_bpp_fail = (float(L_fail) / float(N_fail)) if isinstance(L_fail, int) and isinstance(N_fail, int) and N_fail else ""
        # Pull worker return information from the exception text when available.
        worker_rc_fail = ""
        worker_stderr_fail = ""
        err_text = str(exc)
        if "pySTC worker failed rc=" in err_text:
            try:
                worker_rc_fail = err_text.split("pySTC worker failed rc=", 1)[1].split(":", 1)[0].strip()
                worker_stderr_fail = err_text.split(":", 1)[1].strip()[-2000:]
            except Exception:
                worker_stderr_fail = err_text[-2000:]
        row.update({
            "run_id": run_id,
            "timestamp": now_iso(),
            "image_id": image_id,
            "method": method,
            "payload_bpp": payload_bpp,
            "payload_length_L": L_fail,
            "cover_path": str(cover_path),
            "stego_path": "",
            "cover_sha256": sha256_file(cover_path) if cover_path.exists() else "",
            "stego_sha256": "",
            "height": H_fail, "width": W_fail, "num_pixels": N_fail, "num_selected_pixels": "",
            "num_modified_pixels": "", "change_rate": "", "actual_payload_bpp": actual_payload_bpp_fail,
            "embedding_efficiency": "", "coding_loss": "", "stc_used": False,
            "stc_mode": "failed",
            "stc_backend_requested": getattr(args, "stc_backend", "proxy"),
            "stc_backend_actual": "failed",
            "pystc_available": pystc is not None,
            "stc_seed": stc_seed_fail,
            "stc_message_bytes": int((L_fail + 7)//8) if isinstance(L_fail, int) else "",
            "stc_padded_bits": int((-L_fail) % 8) if isinstance(L_fail, int) else "",
            "stc_safe_subprocess": getattr(args, "stc_safe_subprocess", False),
            "worker_returncode": worker_rc_fail,
            "worker_stderr": worker_stderr_fail,
            "mse": "", "psnr_db": "", "ssim": "",
            "mean_cost": mean_cost_fail, "median_cost": median_cost_fail, "sum_cost": sum_cost_fail, "mean_cost_modified": "",
            "modified_in_high_variance_pct": "",
            "embed_time_sec": time.perf_counter() - t_embed0,
            "extract_time_sec": "", "success_extract": 0, "ber": "",
            "cpu_user_time_sec": cpu1["cpu_user_time_sec"],
            "cpu_system_time_sec": cpu1["cpu_system_time_sec"],
            "process_rss_mb": process_rss_mb(),
            "peak_rss_mb": peak_rss_mb(),
            **ram, **gpu,
            "gamma1_path": str(args.gamma1) if args.gamma1 else "",
            "gamma2_path": str(args.gamma2) if args.gamma2 else "",
            "gamma1_sha256": sha256_file(Path(args.gamma1)) if args.gamma1 and Path(args.gamma1).exists() else "",
            "gamma2_sha256": sha256_file(Path(args.gamma2)) if args.gamma2 and Path(args.gamma2).exists() else "",
            "seed": args.seed,
            "algo_params": "{}",
            "ok": ok,
            "error": err,
        })

    append_csv(logs_dir / "embed_log_mipod.csv", CSV_HEADER, row)
    append_jsonl(logs_dir / "embed_log_mipod.jsonl", row)
    return row


def summarize(rows: List[Dict[str, object]], out_root: Path, run_id: str) -> None:
    logs = out_root / "logs"
    by_payload: Dict[float, List[Dict[str, object]]] = {}
    for r in rows:
        try:
            p = float(r["payload_bpp"])
        except Exception:
            continue
        by_payload.setdefault(p, []).append(r)

    def fvals(rs, key):
        vals = []
        for x in rs:
            try:
                v = float(x[key])
                if math.isfinite(v): vals.append(v)
            except Exception:
                pass
        return np.array(vals, dtype=float)

    for payload, rs in sorted(by_payload.items()):
        n = len(rs)
        ok = np.array([int(r.get("ok", 0)) for r in rs], dtype=int)
        row = {
            "run_id": run_id,
            "method": "MiPOD",
            "payload_bpp": payload,
            "n": n,
            "fail_rate": float(np.mean(ok == 0)) if n else float("nan"),
            "mean_embed_time_sec": float(np.mean(fvals(rs, "embed_time_sec"))) if fvals(rs, "embed_time_sec").size else float("nan"),
            "mean_extract_time_sec": float(np.mean(fvals(rs, "extract_time_sec"))) if fvals(rs, "extract_time_sec").size else float("nan"),
            "mean_ber": float(np.mean(fvals(rs, "ber"))) if fvals(rs, "ber").size else float("nan"),
            "success_rate": float(np.mean(fvals(rs, "success_extract"))) if fvals(rs, "success_extract").size else float("nan"),
            "mean_change_rate": float(np.mean(fvals(rs, "change_rate"))) if fvals(rs, "change_rate").size else float("nan"),
            "mean_embedding_efficiency": float(np.mean(fvals(rs, "embedding_efficiency"))) if fvals(rs, "embedding_efficiency").size else float("nan"),
            "mean_coding_loss": float(np.mean(fvals(rs, "coding_loss"))) if fvals(rs, "coding_loss").size else float("nan"),
            "mean_mse": float(np.mean(fvals(rs, "mse"))) if fvals(rs, "mse").size else float("nan"),
            "median_psnr_db": float(np.median(fvals(rs, "psnr_db"))) if fvals(rs, "psnr_db").size else float("nan"),
            "mean_ssim": float(np.mean(fvals(rs, "ssim"))) if fvals(rs, "ssim").size else float("nan"),
            "out_dir": str(out_root / f"mipod_{payload:.3f}"),
        }
        append_csv(logs / "embed_summary_mipod.csv", SUMMARY_HEADER, row)


# ------------------------------- CLI -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MiPOD baseline + pySTC adaptive-cost embedding for BOSSBase.")
    p.add_argument("--boss-root", default="/home/omego/Downloads/BOSSbase", help="BOSSBase root containing .pgm images")
    p.add_argument("--out-root", default=None, help="Output root for stego images and logs")
    p.add_argument("--gamma1", default="", help="Optional only for log-schema traceability; not used for payload masking")
    p.add_argument("--gamma2", default="", help="Optional only for log-schema traceability; not used for payload masking")
    p.add_argument("--payloads", default="0.1,0.2,0.4", help="Comma-separated payloads in bpp")
    p.add_argument("--max-images", type=int, default=0, help="Process only first N images for testing; 0 means all")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--master-secret", default="change-this-demo-secret", help="Used only to derive deterministic evaluation bits")
    p.add_argument("--deterministic-payload", action="store_true", help="Use HMAC-derived payload bits instead of numpy RNG")
    p.add_argument("--wiener-size", type=int, default=3, help="Wiener filter size for MiPOD content suppression")
    p.add_argument("--variance-window", type=int, default=9, help="Local window size for residual variance estimation")
    p.add_argument("--sigma-floor", type=float, default=1e-4, help="Minimum residual variance for numerical stability")
    p.add_argument("--wet-cost", type=float, default=1e10, help="Wet cost used to prohibit boundary/unsafe changes")
    p.add_argument("--high-variance-percentile", type=float, default=75.0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--progress", choices=["auto", "bar", "none"], default="auto")
    p.add_argument("--stc-backend", choices=["auto", "pystc", "proxy"], default="proxy",
                   help="Embedding backend: proxy, pystc, or auto fallback to proxy if pySTC fails.")
    p.add_argument("--stc-safe-subprocess", action="store_true",
                   help="Run pySTC in a subprocess per image/payload so C++ aborts are logged instead of killing the full run.")
    p.add_argument("--stc-timeout-sec", type=float, default=120.0,
                   help="Timeout for each pySTC subprocess job when --stc-safe-subprocess is enabled.")
    # Hidden worker arguments used internally by --stc-safe-subprocess.
    p.add_argument("--_pystc-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-in", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    if getattr(args, "_pystc_worker", False):
        if not args.worker_in or not args.worker_out:
            raise ValueError("Worker mode requires --worker-in and --worker-out")
        raise SystemExit(pystc_worker_main(args.worker_in, args.worker_out))

    if not getattr(args, "out_root", None):
        raise SystemExit("error: --out-root is required unless running internal pySTC worker mode")
    boss_root = Path(args.boss_root)
    out_root = Path(args.out_root)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gamma1_path = Path(args.gamma1) if args.gamma1 else None
    gamma2_path = Path(args.gamma2) if args.gamma2 else None
    if not boss_root.exists():
        raise FileNotFoundError(f"BOSSBase root not found: {boss_root}")

    run_id = args.run_id or f"mipod-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    payloads = [float(x.strip()) for x in args.payloads.split(",") if x.strip()]
    paths = discover_images(boss_root, max_images=args.max_images if args.max_images > 0 else None)
    if not paths:
        raise RuntimeError(f"No image files found under {boss_root}")

    gamma1_bits = load_cover_stream_bits(gamma1_path) if gamma1_path and gamma1_path.exists() else np.zeros(0, dtype=np.uint8)
    gamma2_bits = load_cover_stream_bits(gamma2_path) if gamma2_path and gamma2_path.exists() else np.zeros(0, dtype=np.uint8)
    master_key = args.master_secret.encode("utf-8")

    jobs = [(p, payload) for payload in payloads for p in paths]
    rows = []
    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] images={len(paths)}, payloads={payloads}, jobs={len(jobs)}")
    print(f"[INFO] optional gamma paths are for log-schema traceability only; gamma bits are NOT used for baseline payload masking. gamma1_bits={gamma1_bits.size}, gamma2_bits={gamma2_bits.size}")
    print(f"[INFO] stc_backend={args.stc_backend}, safe_subprocess={args.stc_safe_subprocess}, pystc_available={pystc is not None}")

    for cover_path, payload in iter_progress(jobs, args.progress):
        row = process_one(cover_path, payload, args, run_id, gamma1_bits, gamma2_bits, master_key, logs_dir)
        rows.append(row)

    summarize(rows, out_root, run_id)
    print(f"[OK] Completed MiPOD run_id={run_id}")
    print(f"Logs:\n  - {logs_dir/'embed_log_mipod.csv'}\n  - {logs_dir/'embed_log_mipod.jsonl'}\n  - {logs_dir/'embed_summary_mipod.csv'}")


if __name__ == "__main__":
    main()
