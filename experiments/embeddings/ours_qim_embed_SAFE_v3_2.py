#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ours_qim_embed_SAFE_v3_2.py — Ours-QIM-v3.2 complex-feature-aware QIM hybrid masking + conservative adaptive embedding for BOSSBase.

This script adapts the structure of the uploaded S-UNIWARD embedding script for the
Ours-QIM variant discussed in the manuscript. It simulates the C3 stego-object
embedding stage offline; it does not require a physical channel.

Pipeline per cover image and payload bpp:
  1) L = round(payload_bpp * H * W)
  2) m <- random payload bits of length L
  3) gamma1, gamma2 <- keyed/generated cover-stream bits, expanded/truncated to L
  4) rho_dit <- Dither(k_stego, psi(o)) as a cover-conditioned bitstream of length L
  5) b_qim = m XOR gamma1 XOR gamma2 XOR rho_dit
  6) Compute a reference-form S-UNIWARD spatial distortion using Daubechies directional residuals.
  7) Conservatively modulate the base costs using rho_i^ours = rho_i^base * g(psi_i(o), rho_dit_i),
     where psi_i(o) is residual-energy based and g is normalised to preserve the global cost scale.
  8) Optionally derive separate +1/-1 costs from a cover-conditioned predictor/dither preference.
  9) Embed b_qim with pySTC or the adaptive-cost proxy using the resulting directional costs.

Important note on STC:
  Publication runs should use --stc-backend pystc --stc-safe-subprocess. The proxy
  remains available only for smoke tests. pySTC receives separate rho(+1) and rho(-1)
  arrays, while the proxy mirrors the same directional-cost ordering. The logged
  coding_loss field is a cost-based proxy rather than the exact STC rate-distortion gap.

Outputs:
  out-root/
    ours_qim_0.100/*.pgm
    ours_qim_0.200/*.pgm
    ours_qim_0.400/*.pgm
    logs/embed_log_ours_qim.csv
    logs/embed_log_ours_qim.jsonl
    logs/embed_summary_ours_qim.csv

Author: generated for Hybrid Multichannel Steganography evaluation.

Scientific naming note:
  This remains a QIM-style cover-conditioned dither-mask method, not canonical sample-lattice QIM.
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
    from scipy.signal import convolve2d as _convolve2d
except Exception:
    _convolve2d = None

try:
    from scipy.ndimage import uniform_filter as _uniform_filter
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    from scipy.ndimage import sobel as _sobel
except Exception:
    _uniform_filter = None
    _gaussian_filter = None
    _sobel = None

try:
    from skimage.segmentation import slic as _slic
except Exception:
    _slic = None

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

try:
    import torch
except Exception:
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

# Daubechies-8 decomposition high-pass filter used by the reference
# S-UNIWARD implementation. The low-pass filter is obtained by quadrature
# mirror construction. Values follow the commonly distributed DDE/Binghamton
# S-UNIWARD Matlab implementation.
_SUNI_HPDF = np.array([
    -0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837,
     0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266,
     0.0173693010, -0.0440882539, -0.0139810279, 0.0087460940,
     0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768,
], dtype=np.float64)
_SUNI_LPDF = ((-1.0) ** np.arange(_SUNI_HPDF.size)) * _SUNI_HPDF[::-1]
_SUNI_FILTERS = (
    np.outer(_SUNI_LPDF, _SUNI_HPDF),
    np.outer(_SUNI_HPDF, _SUNI_LPDF),
    np.outer(_SUNI_HPDF, _SUNI_HPDF),
)


def _conv2_same_symm(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution with symmetric boundary handling."""
    if _convolve2d is None:
        raise RuntimeError(
            "scipy is required for the reference-form S-UNIWARD cost map; "
            "install with: pip install scipy"
        )
    return _convolve2d(
        x.astype(np.float64), kernel.astype(np.float64),
        mode="same", boundary="symm"
    )


def local_mean_map(img_u8: np.ndarray) -> np.ndarray:
    """Four-neighbour predictor used only for optional directional-cost modulation."""
    x = img_u8.astype(np.float64)
    p = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    pred = (
        p[1:-1, :-2] + p[1:-1, 2:] + p[:-2, 1:-1] + p[2:, 1:-1]
    ) / 4.0
    return pred.astype(np.float32)


def local_variance_map(img_u8: np.ndarray) -> np.ndarray:
    """3x3 local variance retained for compatibility and high-variance logging."""
    x = img_u8.astype(np.float32)
    pads = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    vals = []
    for dy in range(3):
        for dx in range(3):
            vals.append(pads[dy:dy + x.shape[0], dx:dx + x.shape[1]])
    stack = np.stack(vals, axis=0)
    return np.var(stack, axis=0).astype(np.float32)


def suniward_reference_costs(
    img_u8: np.ndarray,
    sigma: float = 1.0,
    wet_cost: float = 1e13,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reference-form spatial S-UNIWARD additive costs.

    The cost is the sum, over the three directional Daubechies residuals, of
    the impact of a unit pixel change weighted by 1/(|residual| + sigma).
    Standard S-UNIWARD is symmetric for +1/-1 changes except at image bounds;
    therefore rhoP1 and rhoM1 initially differ only through wet costs.

    Returns:
      rho_p1, rho_m1, residual_energy
    """
    if sigma <= 0:
        raise ValueError("--suniward-sigma must be > 0")
    if wet_cost <= 0:
        raise ValueError("--wet-cost must be > 0")

    x = img_u8.astype(np.float64)
    rho = np.zeros_like(x, dtype=np.float64)
    residual_energy = np.zeros_like(x, dtype=np.float64)

    for kernel in _SUNI_FILTERS:
        residual = _conv2_same_symm(x, kernel)
        residual_energy += np.abs(residual)
        suitability = 1.0 / (np.abs(residual) + float(sigma))
        # The additive impact of changing a pixel is obtained by convolving
        # suitability with the rotated absolute filter.
        rho += _conv2_same_symm(suitability, np.rot90(np.abs(kernel), 2))

    finite = np.isfinite(rho) & (rho >= 0)
    if not np.any(finite):
        raise RuntimeError("S-UNIWARD cost computation produced no finite costs")
    replacement = float(np.median(rho[finite]))
    rho = np.nan_to_num(rho, nan=replacement, posinf=float(wet_cost), neginf=replacement)
    rho = np.clip(rho, 0.0, float(wet_cost))

    rho_p1 = rho.copy()
    rho_m1 = rho.copy()
    rho_p1[img_u8 >= 255] = float(wet_cost)
    rho_m1[img_u8 <= 0] = float(wet_cost)
    return rho_p1.astype(np.float64), rho_m1.astype(np.float64), residual_energy.astype(np.float32)


def suniward_like_cost_map(img_u8: np.ndarray, wavelet: str = "db8", levels: int = 2,
                           eps: float = 1e-3, mode: str = "periodization") -> np.ndarray:
    """
    Backward-compatible wrapper. New publication runs should call
    suniward_reference_costs() directly. The legacy arguments are accepted so
    older commands do not break.
    """
    del wavelet, levels, mode
    sigma = max(float(eps), 1e-12)
    rho_p1, rho_m1, _ = suniward_reference_costs(img_u8, sigma=sigma)
    return np.minimum(rho_p1, rho_m1).astype(np.float32)



def mipod_reference_costs(
    img_u8: np.ndarray,
    variance_kernel: int = 7,
    variance_floor: float = 0.01,
    cost_power: float = 1.0,
    smooth_kernel: int = 7,
    wet_cost: float = 1e13,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Practical MiPOD-inspired spatial costs for smoke-test comparison.

    This is a dependency-light approximation of the MiPOD modelling idea: estimate
    local noise variance from denoising residuals, assign lower costs to locations
    with higher estimated variance/texture, and optionally smooth the Fisher-like
    cost map. It is intended as an alternative base-cost model for Ours-QIM, while
    STC/pySTC still performs the actual embedding.

    Returns:
      rho_p1, rho_m1, residual_energy
    """
    if variance_kernel <= 1:
        variance_kernel = 3
    if variance_kernel % 2 == 0:
        variance_kernel += 1
    if smooth_kernel <= 1:
        smooth_kernel = 1
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1
    if variance_floor <= 0:
        raise ValueError("--mipod-variance-floor must be > 0")
    if cost_power <= 0:
        raise ValueError("--mipod-cost-power must be > 0")

    x = img_u8.astype(np.float64)
    # Denoising predictor: Gaussian when available; otherwise reflective mean.
    if _gaussian_filter is not None:
        den = _gaussian_filter(x, sigma=1.0, mode="reflect")
    else:
        den = _mean_filter(x, 3).astype(np.float64)
    residual = x - den
    residual_energy = np.abs(residual).astype(np.float32)

    # MiPOD-style local variance proxy from denoising residual power, blended
    # with neighbourhood variance so highly textured/edge areas are not over-penalised.
    res_var = _mean_filter((residual * residual).astype(np.float64), int(variance_kernel)).astype(np.float64)
    loc_var = local_variance_map(img_u8).astype(np.float64)
    loc_var_norm = robust_normalize_feature_map(np.log1p(loc_var)).astype(np.float64)
    res_var_norm = robust_normalize_feature_map(np.log1p(res_var)).astype(np.float64)
    sigma2_norm = 0.70 * res_var_norm + 0.30 * loc_var_norm
    sigma2 = np.maximum(sigma2_norm, float(variance_floor))

    # Fisher-like detectability proxy: smooth regions have low variance and
    # therefore receive high cost; complex/noisy regions receive lower cost.
    rho = 1.0 / np.power(sigma2 + float(variance_floor), float(cost_power))
    if smooth_kernel > 1:
        rho = _mean_filter(rho, int(smooth_kernel)).astype(np.float64)

    finite = np.isfinite(rho) & (rho > 0)
    if not np.any(finite):
        raise RuntimeError("MiPOD-like cost computation produced no finite costs")
    lo, hi = np.percentile(rho[finite], [0.1, 99.9])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        rho = np.clip(rho, lo, hi)
    med = float(np.median(rho[finite]))
    if np.isfinite(med) and med > 0:
        rho = rho / med
    rho = np.nan_to_num(rho, nan=1.0, posinf=float(wet_cost), neginf=1.0)
    rho = np.clip(rho, 0.0, float(wet_cost))

    rho_p1 = rho.copy()
    rho_m1 = rho.copy()
    rho_p1[img_u8 >= 255] = float(wet_cost)
    rho_m1[img_u8 <= 0] = float(wet_cost)
    return rho_p1.astype(np.float64), rho_m1.astype(np.float64), residual_energy.astype(np.float32)


def _scale_cost_to_reference(rho: np.ndarray, reference: np.ndarray, wet_cost: float) -> np.ndarray:
    """Scale a cost map so its finite median matches a reference cost map."""
    r = rho.astype(np.float64).copy()
    ref = reference.astype(np.float64)
    valid_r = np.isfinite(r) & (r > 0) & (r < float(wet_cost) * 0.5)
    valid_ref = np.isfinite(ref) & (ref > 0) & (ref < float(wet_cost) * 0.5)
    if np.any(valid_r) and np.any(valid_ref):
        mr = float(np.median(r[valid_r]))
        mf = float(np.median(ref[valid_ref]))
        if np.isfinite(mr) and np.isfinite(mf) and mr > 0 and mf > 0:
            r *= (mf / mr)
    return np.clip(np.nan_to_num(r, nan=float(wet_cost), posinf=float(wet_cost), neginf=float(wet_cost)), 0.0, float(wet_cost))


def base_cost_label(args) -> str:
    mode = getattr(args, "base_cost", "suniward")
    if mode == "suniward":
        return "reference_form_suniward_spatial"
    if mode == "mipod":
        return "mipod_like_variance_fisher_spatial"
    if mode == "fused":
        return "fused_suniward_mipod_spatial"
    return str(mode)


def compute_base_costs(
    img_u8: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Select the base distortion model used by Ours-QIM.

    --base-cost suniward: original V3.1 reference-form S-UNIWARD base cost.
    --base-cost mipod: MiPOD-inspired variance/Fisher base cost.
    --base-cost fused: geometric fusion of S-UNIWARD and MiPOD-like costs.
    """
    wet = float(args.wet_cost)
    s_p1, s_m1, s_res = suniward_reference_costs(img_u8, sigma=args.suniward_sigma, wet_cost=wet)
    mode = getattr(args, "base_cost", "suniward")
    stats = {
        "base_cost_mode": mode,
        "base_cost_type": base_cost_label(args),
        "fused_suniward_weight": float(getattr(args, "fused_suniward_weight", 0.5)),
        "mipod_variance_kernel": int(getattr(args, "mipod_variance_kernel", 7)),
        "mipod_smooth_kernel": int(getattr(args, "mipod_smooth_kernel", 7)),
        "mipod_variance_floor": float(getattr(args, "mipod_variance_floor", 0.01)),
        "mipod_cost_power": float(getattr(args, "mipod_cost_power", 1.0)),
    }
    if mode == "suniward":
        return s_p1, s_m1, s_res, stats

    m_p1, m_m1, m_res = mipod_reference_costs(
        img_u8,
        variance_kernel=int(args.mipod_variance_kernel),
        variance_floor=float(args.mipod_variance_floor),
        cost_power=float(args.mipod_cost_power),
        smooth_kernel=int(args.mipod_smooth_kernel),
        wet_cost=wet,
    )
    # Bring MiPOD-like costs onto the same median scale as S-UNIWARD so pySTC
    # behaviour remains stable and fused costs are meaningful.
    s_min = np.minimum(s_p1, s_m1)
    m_p1 = _scale_cost_to_reference(m_p1, s_min, wet)
    m_m1 = _scale_cost_to_reference(m_m1, s_min, wet)

    if mode == "mipod":
        return m_p1, m_m1, m_res, stats
    if mode == "fused":
        alpha = float(args.fused_suniward_weight)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("--fused-suniward-weight must be in [0,1]")
        eps = 1e-12
        f_p1 = np.exp(alpha * np.log(np.maximum(s_p1, eps)) + (1.0 - alpha) * np.log(np.maximum(m_p1, eps)))
        f_m1 = np.exp(alpha * np.log(np.maximum(s_m1, eps)) + (1.0 - alpha) * np.log(np.maximum(m_m1, eps)))
        f_p1[img_u8 >= 255] = wet
        f_m1[img_u8 <= 0] = wet
        f_res = (alpha * robust_normalize_feature_map(np.log1p(s_res)).astype(np.float64)
                 + (1.0 - alpha) * robust_normalize_feature_map(np.log1p(m_res)).astype(np.float64))
        return f_p1.astype(np.float64), f_m1.astype(np.float64), f_res.astype(np.float32), stats
    raise ValueError(f"Unknown --base-cost: {mode}")


# ------------------------------- Embedding -------------------------------

def keyed_noise(shape: Tuple[int, int], key: bytes, label: str) -> np.ndarray:
    n = shape[0] * shape[1]
    raw = hmac_bytes(key, f"noise|{label}".encode(), n * 8)
    vals = np.frombuffer(raw, dtype=">u8").astype(np.float64)
    vals = vals / float(2 ** 64 - 1)
    return vals.reshape(shape)


def robust_normalize_feature_map(x: np.ndarray, p_low: float = 5.0, p_high: float = 95.0) -> np.ndarray:
    """Robustly normalise a public cover feature map psi(o) to [0,1]."""
    y = x.astype(np.float64)
    lo, hi = np.percentile(y, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(y)), float(np.max(y))
    if hi <= lo:
        return np.zeros_like(y, dtype=np.float32)
    y = (y - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)




def _mean_filter(x: np.ndarray, size: int = 3) -> np.ndarray:
    """Small reflective mean filter with a NumPy fallback."""
    size = int(size)
    if size <= 1:
        return x.astype(np.float32, copy=True)
    if _uniform_filter is not None:
        return _uniform_filter(x.astype(np.float64), size=size, mode="reflect").astype(np.float32)
    pad = size // 2
    y = np.pad(x.astype(np.float64), ((pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(x, dtype=np.float64)
    for dy in range(size):
        for dx in range(size):
            out += y[dy:dy + x.shape[0], dx:dx + x.shape[1]]
    return (out / float(size * size)).astype(np.float32)


def fuzzy_edge_confidence_map(img_u8: np.ndarray) -> np.ndarray:
    """
    Lightweight fuzzy/Sobel edge confidence in [0,1]. It approximates the
    fuzzy edge idea used by MaxMiPOD-style methods while remaining fast and
    dependency-light for smoke tests.
    """
    x = img_u8.astype(np.float64) / 255.0
    if _sobel is not None:
        gx = _sobel(x, axis=1, mode="reflect")
        gy = _sobel(x, axis=0, mode="reflect")
    else:
        gx = _conv2_same_symm(x, np.array([[-1.0, 0.0, 1.0]], dtype=np.float64))
        gy = _conv2_same_symm(x, np.array([[-1.0], [0.0], [1.0]], dtype=np.float64))
    mag = np.sqrt(gx * gx + gy * gy)
    edge = robust_normalize_feature_map(mag, 5.0, 99.0).astype(np.float32)
    # Fuzzy membership: weak gradients remain low, strong/continuous edges approach 1.
    return np.clip(edge * edge * (3.0 - 2.0 * edge), 0.0, 1.0).astype(np.float32)


def slight_unsharp_image(img_u8: np.ndarray, amount: float = 0.15, sigma: float = 1.0) -> np.ndarray:
    """Use sharpening as a feature-extraction microscope; never output it as stego."""
    x = img_u8.astype(np.float64)
    if _gaussian_filter is not None:
        blur = _gaussian_filter(x, sigma=float(sigma), mode="reflect")
    else:
        blur = _mean_filter(x, 3).astype(np.float64)
    sharp = x + float(amount) * (x - blur)
    return np.clip(sharp, 0.0, 255.0).astype(np.float32)


def selected_srm_residual_energy(img_u8: np.ndarray) -> np.ndarray:
    """
    Compact SRM-like high-pass residual energy. This is not a full SRM extractor;
    it is a small set of directional residual filters used only to score texture.
    """
    x = img_u8.astype(np.float64)
    filters = [
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float64),
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float64),
        np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float64),
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64),
        np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64) / 2.0,
    ]
    energy = np.zeros_like(x, dtype=np.float64)
    for h in filters:
        energy += np.abs(_conv2_same_symm(x, h))
    return energy.astype(np.float32)


def local_entropy_proxy(img_u8: np.ndarray) -> np.ndarray:
    """Cheap entropy/texture proxy based on local absolute deviation."""
    x = img_u8.astype(np.float64)
    mean = _mean_filter(x, 5).astype(np.float64)
    mad = _mean_filter(np.abs(x - mean), 5).astype(np.float64)
    return mad.astype(np.float32)


def blockwise_segments(shape: Tuple[int, int], target_segments: int) -> np.ndarray:
    """Fallback segmentation when skimage.slic is unavailable."""
    h, w = shape
    target_segments = max(1, int(target_segments))
    grid = int(max(1, round(math.sqrt(target_segments))))
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    seg = np.zeros((h, w), dtype=np.int32)
    label = 1
    for yi in range(grid):
        for xi in range(grid):
            seg[ys[yi]:ys[yi+1], xs[xi]:xs[xi+1]] = label
            label += 1
    return seg


def build_region_mask(psi_complex: np.ndarray, img_u8: np.ndarray, args) -> Tuple[np.ndarray, Dict[str, object]]:
    """SLIC/region-wise selection for the clustering rule."""
    if not getattr(args, "cf_use_slic", False):
        mask = np.ones_like(psi_complex, dtype=bool)
        return mask, {
            "slic_enabled": False,
            "slic_available": _slic is not None,
            "slic_fallback": "disabled",
            "slic_num_regions": 0,
            "slic_selected_region_pct": 100.0,
            "region_threshold_pct": float(getattr(args, "region_threshold", 70.0)),
        }
    try:
        if _slic is None:
            segments = blockwise_segments(psi_complex.shape, int(args.slic_segments))
            fallback = "blockwise"
        else:
            # SLIC expects float images. channel_axis=None is required for grayscale.
            segments = _slic(
                img_u8.astype(np.float32) / 255.0,
                n_segments=int(args.slic_segments),
                compactness=float(args.slic_compactness),
                sigma=float(args.slic_sigma),
                start_label=1,
                channel_axis=None,
            ).astype(np.int32)
            fallback = "none"
        labels = np.unique(segments)
        region_scores = np.zeros(labels.max() + 1, dtype=np.float64)
        for lab in labels:
            region_scores[lab] = float(np.mean(psi_complex[segments == lab]))
        valid_scores = region_scores[labels]
        thr = float(np.percentile(valid_scores, float(args.region_threshold)))
        selected_regions = set(int(lab) for lab in labels if region_scores[lab] >= thr)
        mask = np.isin(segments, list(selected_regions))
        if not np.any(mask):
            mask = psi_complex >= np.percentile(psi_complex, float(args.region_threshold))
        return mask.astype(bool), {
            "slic_enabled": True,
            "slic_available": _slic is not None,
            "slic_fallback": fallback,
            "slic_num_regions": int(labels.size),
            "slic_selected_region_pct": float(np.mean(mask) * 100.0),
            "region_threshold_pct": float(args.region_threshold),
        }
    except Exception as exc:
        mask = psi_complex >= np.percentile(psi_complex, float(args.region_threshold))
        return mask.astype(bool), {
            "slic_enabled": False,
            "slic_available": _slic is not None,
            "slic_fallback": f"error:{type(exc).__name__}",
            "slic_num_regions": 0,
            "slic_selected_region_pct": float(np.mean(mask) * 100.0),
            "region_threshold_pct": float(args.region_threshold),
        }


def compute_complex_feature_map(
    img_u8: np.ndarray,
    suniward_residual_energy: np.ndarray,
    args,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    """
    MaxMiPOD/CAAS-inspired complex-feature map. It combines residual energy,
    SRM-like residuals, fuzzy edges, sharpened residuals, local variance, and a
    small entropy proxy. Values are robust-normalised to [0,1].
    """
    var_map = local_variance_map(img_u8)
    edge_map = fuzzy_edge_confidence_map(img_u8)
    sharp = slight_unsharp_image(img_u8, amount=float(args.cf_sharpen_amount), sigma=float(args.cf_sharpen_sigma))
    sharp_residual = np.abs(sharp.astype(np.float64) - _mean_filter(sharp, 3).astype(np.float64)).astype(np.float32)
    srm_energy = selected_srm_residual_energy(img_u8)
    entropy_proxy = local_entropy_proxy(img_u8)

    components = {
        "suniward": robust_normalize_feature_map(np.log1p(suniward_residual_energy)),
        "srm": robust_normalize_feature_map(np.log1p(srm_energy)),
        "edge": robust_normalize_feature_map(edge_map),
        "sharp": robust_normalize_feature_map(np.log1p(sharp_residual)),
        "variance": robust_normalize_feature_map(np.log1p(var_map)),
        "entropy": robust_normalize_feature_map(np.log1p(entropy_proxy)),
    }
    weights = {
        "suniward": float(args.cf_alpha_suniward),
        "srm": float(args.cf_alpha_srm),
        "edge": float(args.cf_alpha_edge),
        "sharp": float(args.cf_alpha_sharp),
        "variance": float(args.cf_alpha_var),
        "entropy": float(args.cf_alpha_entropy),
    }
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        weights["suniward"] = 1.0
        total = 1.0
    psi = np.zeros_like(img_u8, dtype=np.float64)
    for name, comp in components.items():
        psi += max(0.0, weights[name]) * comp.astype(np.float64)
    psi = psi / float(total)
    psi = robust_normalize_feature_map(psi, 2.0, 98.0).astype(np.float32)
    region_mask, region_stats = build_region_mask(psi, img_u8, args)
    stats = {
        "cf_enabled": True,
        "cf_map_type": "suniward+srm+fuzzy_edge+sharp_residual+variance+entropy",
        "cf_psi_complex_mean": float(np.mean(psi)),
        "cf_psi_complex_std": float(np.std(psi)),
        "cf_psi_complex_median": float(np.median(psi)),
        "cf_alpha_suniward": float(args.cf_alpha_suniward),
        "cf_alpha_srm": float(args.cf_alpha_srm),
        "cf_alpha_edge": float(args.cf_alpha_edge),
        "cf_alpha_sharp": float(args.cf_alpha_sharp),
        "cf_alpha_var": float(args.cf_alpha_var),
        "cf_alpha_entropy": float(args.cf_alpha_entropy),
        "cf_sharpen_amount": float(args.cf_sharpen_amount),
        "cf_sharpen_sigma": float(args.cf_sharpen_sigma),
        "fuzzy_edge_mean": float(np.mean(edge_map)),
        "fuzzy_edge_std": float(np.std(edge_map)),
        "fuzzy_edge_selected_pct": float(np.mean(edge_map >= np.percentile(edge_map, 75.0)) * 100.0),
        **region_stats,
    }
    maps = {
        "edge_map": edge_map.astype(np.float32),
        "var_map": var_map.astype(np.float32),
        "srm_energy": srm_energy.astype(np.float32),
        "sharp_residual": sharp_residual.astype(np.float32),
        "region_mask": region_mask.astype(bool),
    }
    return psi, maps, stats


def apply_spreading_clustering_costs(
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    psi_complex: np.ndarray,
    region_mask: np.ndarray,
    cover: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Apply spreading and clustering rules after QIM modulation."""
    p = rho_p1.astype(np.float64).copy()
    m = rho_m1.astype(np.float64).copy()
    wet = float(args.wet_cost)
    # Clustering/region rule: penalise low-complexity regions rather than always wetting them.
    penalty = float(args.cluster_penalty)
    outside = ~region_mask.astype(bool)
    if getattr(args, "cf_wet_outside_regions", False):
        p[outside] = wet
        m[outside] = wet
    elif penalty > 1.0:
        p[outside] *= penalty
        m[outside] *= penalty

    # Complexity-first fine adjustment on top of region selection.
    cdiscount = float(args.cf_complexity_discount)
    spenalty = float(args.cf_smooth_penalty)
    g = 1.0 + spenalty * (1.0 - psi_complex.astype(np.float64)) - cdiscount * psi_complex.astype(np.float64)
    g = np.clip(g, float(args.cf_min_factor), float(args.cf_max_factor))
    p *= g
    m *= g

    # Spreading: smooth finite costs locally to reduce isolated selection-channel spikes.
    k = int(args.cost_smooth_kernel)
    if k > 1:
        wet_p = p >= wet * 0.5
        wet_m = m >= wet * 0.5
        p_s = _mean_filter(np.minimum(p, wet).astype(np.float64), k).astype(np.float64)
        m_s = _mean_filter(np.minimum(m, wet).astype(np.float64), k).astype(np.float64)
        mix = float(args.cost_smooth_mix)
        p = (1.0 - mix) * p + mix * p_s
        m = (1.0 - mix) * m + mix * m_s
        p[wet_p] = wet
        m[wet_m] = wet

    p[cover >= 255] = wet
    m[cover <= 0] = wet
    p = np.clip(np.nan_to_num(p, nan=wet, posinf=wet, neginf=wet), 0.0, wet)
    m = np.clip(np.nan_to_num(m, nan=wet, posinf=wet, neginf=wet), 0.0, wet)
    stats = {
        "cost_smooth_kernel": int(k),
        "cost_smooth_mix": float(args.cost_smooth_mix),
        "cluster_penalty": float(args.cluster_penalty),
        "cf_wet_outside_regions": bool(getattr(args, "cf_wet_outside_regions", False)),
        "cf_complexity_discount": float(args.cf_complexity_discount),
        "cf_smooth_penalty": float(args.cf_smooth_penalty),
        "cf_min_factor": float(args.cf_min_factor),
        "cf_max_factor": float(args.cf_max_factor),
    }
    return p.astype(np.float64), m.astype(np.float64), stats

def cover_feature_digest(psi_norm: np.ndarray) -> bytes:
    """Stable digest of quantised public cover features for cover-conditioned dither."""
    q = np.clip(np.rint(psi_norm.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).digest()


def dither_uniform_map(shape: Tuple[int, int], key: bytes, label: str, psi_digest: bytes) -> np.ndarray:
    """Keyed cover-conditioned dither map U(0,1), conditioned on psi(o)."""
    n = shape[0] * shape[1]
    raw = hmac_bytes(key, b"rho-dit-map|" + label.encode() + b"|" + psi_digest, n * 8)
    vals = np.frombuffer(raw, dtype=">u8").astype(np.float64) / float(2 ** 64 - 1)
    return vals.reshape(shape).astype(np.float32)


def dither_bits(key: bytes, label: str, psi_digest: bytes, nbits: int) -> np.ndarray:
    """Keyed cover-conditioned dither bitstream rho_dit for b_qim."""
    return hmac_bits(key, b"rho-dit-bits|" + label.encode() + b"|" + psi_digest, nbits)


def qim_modulated_cost_map(
    rho_base: np.ndarray,
    psi_norm: np.ndarray,
    rho_dit_u: np.ndarray,
    texture_discount: float,
    smooth_penalty: float,
    dither_strength: float,
    min_factor: float,
    max_factor: float,
    normalize_factor: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement rho_i^ours = rho_i^base * g(psi_i(o), rho_dit_i).

    Compared with v1, the modulation is deliberately conservative and the
    factor is optionally normalised to geometric mean one. This preserves the
    global scale of the base distortion while changing only relative pixel
    preference. The keyed dither term is intentionally small by default.
    """
    if min_factor <= 0 or max_factor <= 0 or min_factor > max_factor:
        raise ValueError("Require 0 < --qim-min-factor <= --qim-max-factor")
    psi = np.clip(psi_norm.astype(np.float64), 0.0, 1.0)
    dit = np.clip(rho_dit_u.astype(np.float64), 0.0, 1.0)

    g_texture = 1.0 + float(smooth_penalty) * (1.0 - psi) - float(texture_discount) * psi
    g_dither = 1.0 + float(dither_strength) * (dit - 0.5)
    g = np.maximum(g_texture * g_dither, 1e-12)

    if normalize_factor:
        # Geometric-mean normalisation is natural for multiplicative costs and
        # avoids changing the global cost scale merely because of modulation.
        gm = float(np.exp(np.mean(np.log(g))))
        if np.isfinite(gm) and gm > 0:
            g = g / gm

    g = np.clip(g, float(min_factor), float(max_factor))
    rho = rho_base.astype(np.float64) * g
    finite = np.isfinite(rho) & (rho >= 0)
    replacement = float(np.median(rho[finite])) if np.any(finite) else 1.0
    rho = np.nan_to_num(rho, nan=replacement, posinf=np.finfo(np.float64).max / 4.0, neginf=replacement)
    return rho.astype(np.float64), g.astype(np.float32)


def qim_directional_costs(
    cover: np.ndarray,
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    psi_norm: np.ndarray,
    rho_dit_u: np.ndarray,
    mode: str,
    strength: float,
    dither_mix: float,
    residual_scale: float,
    wet_cost: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optionally derive separate +1/-1 costs.

    mode='none' preserves standard symmetric S-UNIWARD-style polarity costs.
    mode='predictor-dither' introduces a small cover-conditioned polarity
    preference. The predictor term favours changes that move a pixel toward a
    four-neighbour predictor, while the dither term prevents a fully systematic
    polarity bias. The geometric mean of the two directional multipliers is one,
    so this stage changes direction preference without changing base magnitude.

    Positive direction_score means +1 is preferred; negative means -1.
    """
    if mode not in {"none", "predictor-dither", "adaptive-residual"}:
        raise ValueError(f"Unknown --qim-direction-mode: {mode}")
    if strength < 0:
        raise ValueError("--qim-direction-strength must be >= 0")
    if not (0.0 <= dither_mix <= 1.0):
        raise ValueError("--qim-direction-dither-mix must be in [0,1]")
    if residual_scale <= 0:
        raise ValueError("--qim-direction-residual-scale must be > 0")

    if mode == "none" or strength == 0.0:
        score = np.zeros_like(cover, dtype=np.float32)
        out_p1 = rho_p1.astype(np.float64).copy()
        out_m1 = rho_m1.astype(np.float64).copy()
    else:
        pred = local_mean_map(cover).astype(np.float64)
        residual = cover.astype(np.float64) - pred
        # If residual > 0, a -1 change moves toward the predictor; if residual
        # < 0, a +1 change does. tanh limits the influence of large residuals.
        toward_predictor = -np.tanh(residual / float(residual_scale))
        dither_sign = np.where(rho_dit_u.astype(np.float64) >= 0.5, 1.0, -1.0)

        if mode == "adaptive-residual":
            # Stronger V3.1 mode: estimate which polarity produces the smaller
            # local predictor-residual impact. Positive score means +1 has lower
            # residual impact; negative score means -1 has lower residual impact.
            # This is still a cost preference, not a post-embedding perturbation,
            # so extraction reliability remains governed by STC.
            base_abs = np.abs(residual)
            impact_plus = np.abs((cover.astype(np.float64) + 1.0) - pred) - base_abs
            impact_minus = np.abs((cover.astype(np.float64) - 1.0) - pred) - base_abs
            predictor_prefer = np.tanh((impact_minus - impact_plus) / max(float(residual_scale), 1e-6))

            lap = _conv2_same_symm(cover.astype(np.float64), np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64))
            hp_prefer = -np.tanh(lap / max(float(residual_scale) * 2.0, 1e-6))

            # In smooth regions direction should preserve local residuals strongly;
            # in complex regions we retain some keyed dither to avoid a systematic
            # polarity fingerprint.
            smooth_gate = 1.0 - np.clip(psi_norm.astype(np.float64), 0.0, 1.0)
            texture_gate = np.clip(psi_norm.astype(np.float64), 0.0, 1.0)
            score = (
                0.72 * predictor_prefer * (0.55 + 0.45 * smooth_gate)
                + 0.18 * hp_prefer * (0.50 + 0.50 * smooth_gate)
                + 0.10 * dither_sign * (0.25 + 0.75 * texture_gate)
            )
        else:
            # V2-compatible ablation. Direction preference is most meaningful where
            # local prediction has some structure; in highly textured regions the
            # small dither component dominates rather than forcing shrinkage.
            structure_weight = 1.0 - np.clip(psi_norm.astype(np.float64), 0.0, 1.0)
            score = (
                (1.0 - float(dither_mix)) * toward_predictor * structure_weight
                + float(dither_mix) * dither_sign
            )
        score = np.clip(score, -1.0, 1.0)
        plus_factor = np.exp(-float(strength) * score)
        minus_factor = np.exp(float(strength) * score)
        out_p1 = rho_p1.astype(np.float64) * plus_factor
        out_m1 = rho_m1.astype(np.float64) * minus_factor

    # Preserve wet pixels exactly after all modulation.
    out_p1[cover >= 255] = float(wet_cost)
    out_m1[cover <= 0] = float(wet_cost)
    out_p1 = np.clip(np.nan_to_num(out_p1, nan=float(wet_cost), posinf=float(wet_cost), neginf=float(wet_cost)), 0.0, float(wet_cost))
    out_m1 = np.clip(np.nan_to_num(out_m1, nan=float(wet_cost), posinf=float(wet_cost), neginf=float(wet_cost)), 0.0, float(wet_cost))
    return out_p1.astype(np.float64), out_m1.astype(np.float64), np.asarray(score, dtype=np.float32)


def sha256_bits(bits: np.ndarray) -> str:
    """SHA-256 digest of a bit array, packed into bytes for stable logging."""
    b, _ = bits_to_bytes(np.asarray(bits, dtype=np.uint8).reshape(-1))
    return hashlib.sha256(b).hexdigest()


def safe_mean(x) -> float:
    x = np.asarray(x)
    return float(np.mean(x)) if x.size else float("nan")


def safe_median(x) -> float:
    x = np.asarray(x)
    return float(np.median(x)) if x.size else float("nan")


def safe_sum(x) -> float:
    x = np.asarray(x)
    return float(np.sum(x)) if x.size else float("nan")


def compute_qim_base_stats(
    rho_base_p1: np.ndarray,
    rho_base_m1: np.ndarray,
    rho_mod_p1: np.ndarray,
    rho_mod_m1: np.ndarray,
    qim_g: np.ndarray,
    direction_score: np.ndarray,
    rho_dit_bits: np.ndarray,
    residual_energy: np.ndarray,
    args,
) -> Dict[str, object]:
    """Compute QIM-v2 audit statistics independent of the final modified mask."""
    eps = 1e-12
    base_min = np.minimum(rho_base_p1.astype(np.float64), rho_base_m1.astype(np.float64))
    mod_min = np.minimum(rho_mod_p1.astype(np.float64), rho_mod_m1.astype(np.float64))
    ratio = (mod_min / np.maximum(base_min, eps)).reshape(-1)
    base_flat = base_min.reshape(-1)
    ratio = ratio[np.isfinite(ratio) & (base_flat < float(args.wet_cost) * 0.5)]

    finite_g = qim_g[np.isfinite(qim_g) & (qim_g > 0)]
    g_geomean = float(np.exp(np.mean(np.log(finite_g)))) if finite_g.size else float("nan")
    score = np.asarray(direction_score, dtype=np.float64)

    return {
        "qim_enabled": True,
        "qim_version": "Ours-QIM-v3.2",
        "qim_base_cost_type": "reference_form_suniward_spatial",
        "qim_psi_type": "robust_normalized_directional_residual_energy",
        "qim_dither_source": "HMAC-SHA256(k_stego, psi(o))",
        "qim_dither_sha256": sha256_bits(rho_dit_bits),
        "qim_dither_length": int(rho_dit_bits.size),
        "qim_dither_ones_ratio": float(np.mean(rho_dit_bits)) if rho_dit_bits.size else float("nan"),

        "qim_base_cost_mean": float(np.mean(base_min)),
        "qim_base_cost_median": float(np.median(base_min)),
        "qim_base_cost_sum": float(np.sum(base_min)),
        "qim_modulated_cost_mean": float(np.mean(mod_min)),
        "qim_modulated_cost_median": float(np.median(mod_min)),
        "qim_modulated_cost_sum": float(np.sum(mod_min)),
        "qim_cost_ratio_mean": float(np.mean(ratio)) if ratio.size else float("nan"),
        "qim_cost_ratio_median": float(np.median(ratio)) if ratio.size else float("nan"),
        "qim_cost_ratio_min": float(np.min(ratio)) if ratio.size else float("nan"),
        "qim_cost_ratio_max": float(np.max(ratio)) if ratio.size else float("nan"),

        "qim_texture_alpha": float(args.qim_texture_discount),
        "qim_dither_alpha": float(args.qim_dither_strength),
        "qim_g_mean": float(np.mean(qim_g)),
        "qim_g_median": float(np.median(qim_g)),
        "qim_g_min": float(np.min(qim_g)),
        "qim_g_max": float(np.max(qim_g)),
        "qim_g_geomean": g_geomean,

        "qim_direction_mode": str(args.qim_direction_mode),
        "qim_direction_strength": float(args.qim_direction_strength),
        "qim_direction_dither_mix": float(args.qim_direction_dither_mix),
        "qim_direction_score_mean": float(np.mean(score)),
        "qim_direction_score_std": float(np.std(score)),
        "qim_prefer_plus_pct": float(np.mean(score > 0) * 100.0),
        "qim_prefer_minus_pct": float(np.mean(score < 0) * 100.0),
        "qim_rho_p1_mean": float(np.mean(rho_mod_p1)),
        "qim_rho_m1_mean": float(np.mean(rho_mod_m1)),
        "qim_wet_plus_count": int(np.count_nonzero(rho_mod_p1 >= float(args.wet_cost) * 0.5)),
        "qim_wet_minus_count": int(np.count_nonzero(rho_mod_m1 >= float(args.wet_cost) * 0.5)),
        "qim_residual_energy_mean": float(np.mean(residual_energy)),
        "qim_residual_energy_median": float(np.median(residual_energy)),

        # Filled after embedding.
        "qim_modified_dither_one_pct": "",
        "qim_modified_high_texture_pct": "",
        "qim_modified_plus_pct": "",
        "qim_modified_minus_pct": "",
        "qim_modified_preferred_direction_pct": "",
    }


def add_qim_modified_stats(
    qim_stats: Dict[str, object],
    cover: np.ndarray,
    stego: np.ndarray,
    modified_mask: np.ndarray,
    rho_dit_u: np.ndarray,
    psi_norm: np.ndarray,
    direction_score: np.ndarray,
    high_texture_threshold: float = 0.75,
) -> Dict[str, object]:
    """Add QIM-v2 statistics that depend on the final modified pixels."""
    num_modified = int(np.count_nonzero(modified_mask))
    if num_modified <= 0:
        qim_stats.update({
            "qim_modified_dither_one_pct": 0.0,
            "qim_modified_high_texture_pct": 0.0,
            "qim_modified_plus_pct": 0.0,
            "qim_modified_minus_pct": 0.0,
            "qim_modified_preferred_direction_pct": 0.0,
        })
        return qim_stats

    dither_one_map = rho_dit_u >= 0.5
    high_texture_map = psi_norm >= high_texture_threshold
    delta = stego.astype(np.int16) - cover.astype(np.int16)
    plus = delta > 0
    minus = delta < 0
    preferred = ((direction_score > 0) & plus) | ((direction_score < 0) & minus)

    qim_stats["qim_modified_dither_one_pct"] = float(np.mean(dither_one_map[modified_mask]) * 100.0)
    qim_stats["qim_modified_high_texture_pct"] = float(np.mean(high_texture_map[modified_mask]) * 100.0)
    qim_stats["qim_modified_plus_pct"] = float(np.mean(plus[modified_mask]) * 100.0)
    qim_stats["qim_modified_minus_pct"] = float(np.mean(minus[modified_mask]) * 100.0)
    qim_stats["qim_modified_preferred_direction_pct"] = float(np.mean(preferred[modified_mask]) * 100.0)
    return qim_stats


def empty_qim_stats() -> Dict[str, object]:
    """Default QIM-v2 fields for failure rows where setup did not complete."""
    keys = [
        "qim_enabled", "qim_version", "qim_base_cost_type", "qim_psi_type",
        "qim_dither_source", "qim_dither_sha256", "qim_dither_length",
        "qim_dither_ones_ratio", "qim_base_cost_mean", "qim_base_cost_median",
        "qim_base_cost_sum", "qim_modulated_cost_mean", "qim_modulated_cost_median",
        "qim_modulated_cost_sum", "qim_cost_ratio_mean", "qim_cost_ratio_median",
        "qim_cost_ratio_min", "qim_cost_ratio_max", "qim_texture_alpha",
        "qim_dither_alpha", "qim_g_mean", "qim_g_median", "qim_g_min",
        "qim_g_max", "qim_g_geomean", "qim_direction_mode",
        "qim_direction_strength", "qim_direction_dither_mix",
        "qim_direction_score_mean", "qim_direction_score_std", "qim_prefer_plus_pct",
        "qim_prefer_minus_pct", "qim_rho_p1_mean", "qim_rho_m1_mean",
        "qim_wet_plus_count", "qim_wet_minus_count", "qim_residual_energy_mean",
        "qim_residual_energy_median", "qim_modified_dither_one_pct",
        "qim_modified_high_texture_pct", "qim_modified_plus_pct",
        "qim_modified_minus_pct", "qim_modified_preferred_direction_pct",
    ]
    out = {k: "" for k in keys}
    out["qim_enabled"] = True
    out["qim_version"] = "Ours-QIM-v3.2"
    out["qim_base_cost_type"] = "reference_form_suniward_spatial"
    out["qim_psi_type"] = "robust_normalized_directional_residual_energy"
    out["qim_dither_source"] = "HMAC-SHA256(k_stego, psi(o))"
    return out


def empty_cf_stats() -> Dict[str, object]:
    """Default Ours-QIM-v3 complex-feature fields for failure rows."""
    keys = [
        "cf_enabled", "cf_map_type", "cf_psi_complex_mean", "cf_psi_complex_std",
        "cf_psi_complex_median", "cf_alpha_suniward", "cf_alpha_srm",
        "cf_alpha_edge", "cf_alpha_sharp", "cf_alpha_var", "cf_alpha_entropy",
        "cf_sharpen_amount", "cf_sharpen_sigma", "fuzzy_edge_mean",
        "fuzzy_edge_std", "fuzzy_edge_selected_pct", "fuzzy_edge_modified_pct",
        "slic_enabled", "slic_available", "slic_fallback", "slic_num_regions",
        "slic_selected_region_pct", "region_threshold_pct",
        "modified_in_selected_region_pct", "modified_in_cf_top_quartile_pct",
        "cf_modified_complex_pct", "cf_modified_high_complexity_pct",
        "cf_modified_complex_mean", "cf_modified_complex_median",
        "cf_modified_complex_lift", "cf_complex_threshold_percentile",
        "modified_preferred_direction_pct", "modified_preferred_direction_decisive_pct",
        "qim_direction_decisive_modified_pct",
        "cost_smooth_kernel", "cost_smooth_mix", "cluster_penalty",
        "cf_wet_outside_regions", "cf_complexity_discount", "cf_smooth_penalty",
        "cf_min_factor", "cf_max_factor", "adaptive_direction_enabled",
    ]
    out = {k: "" for k in keys}
    out["cf_enabled"] = True
    out["cf_map_type"] = "suniward+srm+fuzzy_edge+sharp_residual+variance+entropy"
    out["slic_available"] = _slic is not None
    return out


def directional_actual_cost(
    cover: np.ndarray,
    stego: np.ndarray,
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Return total directional distortion and per-pixel realized costs."""
    delta = stego.astype(np.int16) - cover.astype(np.int16)
    realized = np.zeros_like(rho_p1, dtype=np.float64)
    realized[delta > 0] = rho_p1[delta > 0]
    realized[delta < 0] = rho_m1[delta < 0]
    return float(np.sum(realized)), realized


def embed_cost_adaptive_lsb_matching(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    key: bytes,
    label: str,
    wet_cost: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Adaptive-cost LSB-matching proxy with separate +1/-1 costs.

    This is only a smoke-test fallback. It selects low minimum-direction-cost
    pixels, writes payload bits to LSBs, and chooses the cheaper valid polarity.
    Publication results should use pySTC.
    """
    H, W = cover.shape
    N = H * W
    L = int(payload_bits.size)
    if L > N:
        raise ValueError(f"payload length L={L} exceeds image capacity N={N}; reduce bpp or add coding layer")

    min_cost = np.minimum(rho_p1, rho_m1).reshape(-1).astype(np.float64)
    embeddable = min_cost < float(wet_cost) * 0.5
    if int(np.count_nonzero(embeddable)) < L:
        raise ValueError(
            f"payload length L={L} exceeds embeddable pixels={int(np.count_nonzero(embeddable))}"
        )

    noise = keyed_noise((H, W), key, label).reshape(-1)
    score = min_cost + 1e-12 * noise
    score[~embeddable] = float(wet_cost)
    selected_idx = np.argpartition(score, L - 1)[:L] if L > 0 else np.array([], dtype=np.int64)
    selected_idx = selected_idx[np.argsort(score[selected_idx])]

    stego = cover.copy().reshape(-1)
    cover_flat = cover.reshape(-1)
    p1 = rho_p1.reshape(-1)
    m1 = rho_m1.reshape(-1)
    current_lsb = (stego[selected_idx] & 1).astype(np.uint8)
    need_change = current_lsb != payload_bits
    change_idx = selected_idx[need_change]

    for idx in change_idx:
        val = int(stego[idx])
        cp = float(p1[idx]) if val < 255 else float(wet_cost)
        cm = float(m1[idx]) if val > 0 else float(wet_cost)
        if cp >= float(wet_cost) * 0.5 and cm >= float(wet_cost) * 0.5:
            raise RuntimeError(f"selected wet pixel at flat index {int(idx)}")
        if cp < cm:
            stego[idx] = val + 1
        elif cm < cp:
            stego[idx] = val - 1
        else:
            # Deterministic keyed tie-breaker only when directional costs are equal.
            sign = hmac_bits(key, f"sign|{label}|{int(idx)}".encode(), 1)[0]
            stego[idx] = val + 1 if sign == 1 and val < 255 else val - 1

    stego2 = stego.reshape(H, W).astype(np.uint8)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected_idx] = True
    selected_mask = selected_mask.reshape(H, W)
    modified_mask = cover != stego2

    flips = int(np.count_nonzero(modified_mask))
    change_rate = flips / float(N)
    payload_bpp_actual = L / float(N)
    embedding_eff = (L / flips) if flips > 0 else float("inf")

    actual_distortion, _ = directional_actual_cost(cover, stego2, rho_p1, rho_m1)
    if flips > 0:
        eligible_costs = min_cost[embeddable]
        lower = float(np.sum(np.partition(eligible_costs, flips - 1)[:flips]))
        coding_loss = (actual_distortion / lower - 1.0) if lower > 0 else float("nan")
    else:
        lower = 0.0
        coding_loss = 0.0

    stats = {
        "payload_bits": L,
        "capacity_pixels": N,
        "num_selected_pixels": int(np.count_nonzero(embeddable)),
        "num_modified_pixels": flips,
        "change_rate": change_rate,
        "actual_payload_bpp": payload_bpp_actual,
        "embedding_efficiency": embedding_eff,
        "actual_distortion": actual_distortion,
        "oracle_min_distortion_same_flips": lower,
        "coding_loss": coding_loss,
        "stc_used": False,
        "stc_mode": "directional_cost_lsb_matching_proxy",
    }
    return stego2, selected_mask, modified_mask, stats


def extract_lsb(
    stego: np.ndarray,
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    key: bytes,
    label: str,
    L: int,
    wet_cost: float,
) -> np.ndarray:
    """Extraction mirrors the proxy's keyed minimum-direction-cost ordering."""
    H, W = stego.shape
    min_cost = np.minimum(rho_p1, rho_m1).reshape(-1).astype(np.float64)
    embeddable = min_cost < float(wet_cost) * 0.5
    noise = keyed_noise((H, W), key, label).reshape(-1)
    score = min_cost + 1e-12 * noise
    score[~embeddable] = float(wet_cost)
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
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Run pySTC directly in the current Python process. This is fast but unsafe if the C++ layer aborts."""
    if pystc is None:
        raise RuntimeError("pySTC is not available; install with: pip install pystcstego")
    msg_bytes, padded = bits_to_bytes(payload_bits)
    t0 = time.perf_counter()
    stego = pystc.hide(msg_bytes, cover.astype(np.uint8), rho_p1.astype(np.float64), rho_m1.astype(np.float64), int(seed), mx=255, mn=0)
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
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    seed: int,
    timeout_sec: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Run pySTC in a child Python process so C++ aborts do not kill the main experiment."""
    msg_bytes, padded = bits_to_bytes(payload_bits)
    with tempfile.TemporaryDirectory(prefix="ours_qim_pystc_") as td:
        td_path = Path(td)
        inp = td_path / "worker_in.npz"
        outp = td_path / "worker_out.npz"
        np.savez_compressed(
            inp,
            cover=cover.astype(np.uint8),
            rho_p1=rho_p1.astype(np.float64),
            rho_m1=rho_m1.astype(np.float64),
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
    rho_p1 = data["rho_p1"].astype(np.float64)
    rho_m1 = data["rho_m1"].astype(np.float64)
    msg = data["msg_bytes"].astype(np.uint8).tobytes()
    seed = int(data["seed"][0])
    L = int(data["L"][0])
    t0 = time.perf_counter()
    stego = _pystc.hide(msg, cover, rho_p1, rho_m1, seed, mx=255, mn=0)
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
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    key: bytes,
    label: str,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Dispatch embedding to pySTC or the directional-cost proxy."""
    requested = args.stc_backend
    seed = pystc_seed_from_key(key, label)
    pystc_available = pystc is not None

    def proxy(mode_suffix: str = "directional_cost_lsb_matching_proxy"):
        stego, selected_mask, modified_mask, estats = embed_cost_adaptive_lsb_matching(
            cover, payload_bits, rho_p1, rho_m1, key, label, float(args.wet_cost)
        )
        extracted = extract_lsb(
            stego, rho_p1, rho_m1, key, label, int(payload_bits.size), float(args.wet_cost)
        )
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
                stego, extracted, stc_stats = run_pystc_subprocess(
                    cover, payload_bits, rho_p1, rho_m1, seed, args.stc_timeout_sec
                )
            else:
                stego, extracted, stc_stats = run_pystc_direct(
                    cover, payload_bits, rho_p1, rho_m1, seed
                )
        except Exception as exc:
            if requested == "auto":
                stego, extracted, selected_mask, modified_mask, estats = proxy("proxy_after_pystc_failure")
                estats["error"] = f"pySTC failed; used proxy fallback: {type(exc).__name__}: {exc}"
                return stego, extracted, selected_mask, modified_mask, estats
            raise

        modified_mask = cover != stego
        embeddable_mask = np.minimum(rho_p1, rho_m1) < float(args.wet_cost) * 0.5
        selected_mask = embeddable_mask.copy()
        flips = int(np.count_nonzero(modified_mask))
        N = int(cover.size)
        L = int(payload_bits.size)
        actual_distortion, _ = directional_actual_cost(cover, stego, rho_p1, rho_m1)
        eligible_costs = np.minimum(rho_p1, rho_m1)[embeddable_mask].astype(np.float64)
        if flips > 0:
            lower = float(np.sum(np.partition(eligible_costs, flips - 1)[:flips]))
            coding_loss = (actual_distortion / lower - 1.0) if lower > 0 else float("nan")
        else:
            lower = 0.0
            coding_loss = 0.0
        estats = {
            "payload_bits": L,
            "capacity_pixels": N,
            "num_selected_pixels": int(np.count_nonzero(embeddable_mask)),
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
        yield from tqdm(items, desc="Ours-QIM embedding", unit="job")
    else:
        total = len(items)
        step = max(1, total // 20)
        for i, item in enumerate(items, 1):
            if progress != "none" and (i == 1 or i % step == 0 or i == total):
                print(f"[Ours-QIM embedding] {100*i/total:5.1f}% ({i}/{total})", flush=True)
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

    # QIM-specific audit fields
    "qim_enabled",
    "qim_version",
    "qim_base_cost_type",
    "base_cost_mode",
    "fused_suniward_weight",
    "mipod_variance_kernel",
    "mipod_smooth_kernel",
    "mipod_variance_floor",
    "mipod_cost_power",
    "qim_psi_type",
    "qim_dither_source",
    "qim_dither_sha256",
    "qim_dither_length",
    "qim_dither_ones_ratio",
    "qim_base_cost_mean",
    "qim_base_cost_median",
    "qim_base_cost_sum",
    "qim_modulated_cost_mean",
    "qim_modulated_cost_median",
    "qim_modulated_cost_sum",
    "qim_cost_ratio_mean",
    "qim_cost_ratio_median",
    "qim_cost_ratio_min",
    "qim_cost_ratio_max",
    "qim_texture_alpha",
    "qim_dither_alpha",
    "qim_g_mean",
    "qim_g_median",
    "qim_g_min",
    "qim_g_max",
    "qim_g_geomean",
    "qim_direction_mode",
    "qim_direction_strength",
    "qim_direction_dither_mix",
    "qim_direction_score_mean",
    "qim_direction_score_std",
    "qim_prefer_plus_pct",
    "qim_prefer_minus_pct",
    "qim_rho_p1_mean",
    "qim_rho_m1_mean",
    "qim_wet_plus_count",
    "qim_wet_minus_count",
    "qim_residual_energy_mean",
    "qim_residual_energy_median",
    "qim_modified_dither_one_pct",
    "qim_modified_high_texture_pct",
    "qim_modified_plus_pct",
    "qim_modified_minus_pct",
    "qim_modified_preferred_direction_pct",

    # Ours-QIM-v3 complex-feature / region / clustering audit fields
    "cf_enabled",
    "cf_map_type",
    "cf_psi_complex_mean",
    "cf_psi_complex_std",
    "cf_psi_complex_median",
    "cf_alpha_suniward",
    "cf_alpha_srm",
    "cf_alpha_edge",
    "cf_alpha_sharp",
    "cf_alpha_var",
    "cf_alpha_entropy",
    "cf_sharpen_amount",
    "cf_sharpen_sigma",
    "fuzzy_edge_mean",
    "fuzzy_edge_std",
    "fuzzy_edge_selected_pct",
    "fuzzy_edge_modified_pct",
    "slic_enabled",
    "slic_available",
    "slic_fallback",
    "slic_num_regions",
    "slic_selected_region_pct",
    "region_threshold_pct",
    "modified_in_selected_region_pct",
    "modified_in_cf_top_quartile_pct",
    "cf_modified_complex_pct",
    "cf_modified_high_complexity_pct",
    "cf_modified_complex_mean",
    "cf_modified_complex_median",
    "cf_modified_complex_lift",
    "cf_complex_threshold_percentile",
    "modified_preferred_direction_pct",
    "modified_preferred_direction_decisive_pct",
    "qim_direction_decisive_modified_pct",
    "cost_smooth_kernel",
    "cost_smooth_mix",
    "cluster_penalty",
    "cf_wet_outside_regions",
    "cf_complexity_discount",
    "cf_smooth_penalty",
    "cf_min_factor",
    "cf_max_factor",
    "adaptive_direction_enabled",

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
    gamma1_bits_base: np.ndarray,
    gamma2_bits_base: np.ndarray,
    master_key: bytes,
    logs_dir: Path,
) -> Dict[str, object]:
    row: Dict[str, object] = {}
    image_id = cover_path.stem
    method = "Ours-QIM"
    out_dir = Path(args.out_root) / f"ours_qim_{payload_bpp:.3f}"
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

        # Per-image/payload keys/labels.
        label = f"{image_id}|{payload_bpp:.3f}|{args.seed}" 
        payload_key = hmac.new(master_key, f"payload|{label}".encode(), hashlib.sha256).digest()
        gamma_key = hmac.new(master_key, f"gamma|{label}".encode(), hashlib.sha256).digest()
        kstego_key = hmac.new(master_key, f"k-stego|{label}".encode(), hashlib.sha256).digest()

        if args.deterministic_payload:
            m_bits = hmac_bits(payload_key, b"m", L)
        else:
            # Derive a per-image/payload RNG seed so non-deterministic evaluation
            # does not accidentally reuse the same payload for every image.
            rng_seed = int.from_bytes(payload_key[:8], "big")
            m_bits = np.random.default_rng(rng_seed).integers(0, 2, L, dtype=np.uint8)
        g1_bits = expand_stream_bits(gamma1_bits_base, L, gamma_key, "gamma1")
        g2_bits = expand_stream_bits(gamma2_bits_base, L, gamma_key, "gamma2")

        # Select the base distortion model. This does not change the overall
        # Hybrid/Ours-QIM protocol; it only changes the image-domain cost map
        # supplied to STC/pySTC.
        rho_base_p1, rho_base_m1, residual_energy, base_cost_stats = compute_base_costs(
            cover, args
        )
        rho_base_min = np.minimum(rho_base_p1, rho_base_m1)

        # V3: use a complex-feature map for dither conditioning and relative cost
        # modulation. Disable with --no-cf-features to reproduce the simpler V2 map.
        if getattr(args, "no_cf_features", False):
            psi_norm = robust_normalize_feature_map(np.log1p(residual_energy))
            cf_maps = {
                "edge_map": np.zeros_like(psi_norm, dtype=np.float32),
                "var_map": local_variance_map(cover),
                "region_mask": np.ones_like(psi_norm, dtype=bool),
            }
            cf_stats = {
                "cf_enabled": False,
                "cf_map_type": "v2_residual_energy_only",
                "cf_psi_complex_mean": float(np.mean(psi_norm)),
                "cf_psi_complex_std": float(np.std(psi_norm)),
                "cf_psi_complex_median": float(np.median(psi_norm)),
                "cf_alpha_suniward": "", "cf_alpha_srm": "", "cf_alpha_edge": "",
                "cf_alpha_sharp": "", "cf_alpha_var": "", "cf_alpha_entropy": "",
                "cf_sharpen_amount": "", "cf_sharpen_sigma": "",
                "fuzzy_edge_mean": "", "fuzzy_edge_std": "", "fuzzy_edge_selected_pct": "",
                "slic_enabled": False, "slic_available": _slic is not None, "slic_fallback": "disabled",
                "slic_num_regions": 0, "slic_selected_region_pct": 100.0,
                "region_threshold_pct": "",
            }
        else:
            psi_norm, cf_maps, cf_stats = compute_complex_feature_map(cover, residual_energy, args)

        psi_digest = cover_feature_digest(psi_norm)
        rho_dit_bits = dither_bits(kstego_key, label, psi_digest, L)
        rho_dit_u = dither_uniform_map(cover.shape, kstego_key, label, psi_digest)
        b_bits = np.bitwise_xor(np.bitwise_xor(m_bits, g1_bits), np.bitwise_xor(g2_bits, rho_dit_bits)).astype(np.uint8)

        _, qim_g = qim_modulated_cost_map(
            rho_base_min, psi_norm, rho_dit_u,
            texture_discount=args.qim_texture_discount,
            smooth_penalty=args.qim_smooth_penalty,
            dither_strength=args.qim_dither_strength,
            min_factor=args.qim_min_factor,
            max_factor=args.qim_max_factor,
            normalize_factor=not args.no_qim_factor_normalization,
        )
        rho_mod_p1 = rho_base_p1 * qim_g.astype(np.float64)
        rho_mod_m1 = rho_base_m1 * qim_g.astype(np.float64)
        rho_p1, rho_m1, direction_score = qim_directional_costs(
            cover=cover,
            rho_p1=rho_mod_p1,
            rho_m1=rho_mod_m1,
            psi_norm=psi_norm,
            rho_dit_u=rho_dit_u,
            mode=args.qim_direction_mode,
            strength=args.qim_direction_strength,
            dither_mix=args.qim_direction_dither_mix,
            residual_scale=args.qim_direction_residual_scale,
            wet_cost=args.wet_cost,
        )
        if not getattr(args, "no_cf_features", False):
            rho_p1, rho_m1, cf_cost_stats = apply_spreading_clustering_costs(
                rho_p1, rho_m1, psi_norm, cf_maps["region_mask"], cover, args
            )
            cf_stats.update(cf_cost_stats)
        else:
            cf_stats.update({
                "cost_smooth_kernel": 0, "cost_smooth_mix": 0.0, "cluster_penalty": 1.0,
                "cf_wet_outside_regions": False, "cf_complexity_discount": 0.0,
                "cf_smooth_penalty": 0.0, "cf_min_factor": 1.0, "cf_max_factor": 1.0,
            })
        rho_min = np.minimum(rho_p1, rho_m1)

        # QIM-specific log fields before embedding.
        qim_stats = compute_qim_base_stats(
            rho_base_p1=rho_base_p1,
            rho_base_m1=rho_base_m1,
            rho_mod_p1=rho_p1,
            rho_mod_m1=rho_m1,
            qim_g=qim_g,
            direction_score=direction_score,
            rho_dit_bits=rho_dit_bits,
            residual_energy=residual_energy,
            args=args,
        )
        qim_stats["qim_psi_type"] = cf_stats.get("cf_map_type", qim_stats.get("qim_psi_type", ""))
        qim_stats["qim_base_cost_type"] = base_cost_stats.get("base_cost_type", qim_stats.get("qim_base_cost_type", ""))

        var_map = cf_maps.get("var_map", local_variance_map(cover))
        high_var_threshold = float(np.percentile(var_map, args.high_variance_percentile))
        high_var_mask = var_map >= high_var_threshold

        embed_key = hmac.new(master_key, f"embed-qim-v3|{label}".encode(), hashlib.sha256).digest()
        t_backend0 = time.perf_counter()
        stego, extracted, selected_mask, modified_mask, estats = embed_with_backend(
            cover, b_bits, rho_p1, rho_m1, embed_key, label, args
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
        _, realized_cost_map = directional_actual_cost(cover, stego, rho_p1, rho_m1)
        cost_mod = realized_cost_map[modified_mask]
        mean_cost_modified = float(np.mean(cost_mod)) if cost_mod.size else 0.0
        mod_high = float(np.mean(high_var_mask[modified_mask]) * 100.0) if np.count_nonzero(modified_mask) else 0.0
        # V3.1 placement and polarity diagnostics.  These fields make the
        # complex-feature claim auditable and give tune_qim_v3.py a stable
        # cf_modified_complex_pct column to aggregate into
        # mean_cf_modified_complex_pct.
        edge_map = cf_maps.get("edge_map", np.zeros_like(cover, dtype=np.float32))
        region_mask = cf_maps.get("region_mask", np.ones_like(cover, dtype=bool))
        complex_thr_pct = float(getattr(args, "cf_complex_threshold_percentile", 75.0))
        complex_thr = float(np.percentile(psi_norm, complex_thr_pct))
        complex_mask = psi_norm >= complex_thr
        edge_thr = float(np.percentile(edge_map, 75.0)) if edge_map.size else 0.0
        edge_complex_mask = edge_map >= edge_thr
        decisive_mask = np.abs(direction_score) >= 1e-6

        if np.count_nonzero(modified_mask):
            modified_complex_values = psi_norm[modified_mask].astype(np.float64)
            cf_complex_pct = float(np.mean(complex_mask[modified_mask]) * 100.0)
            cf_stats["fuzzy_edge_modified_pct"] = float(np.mean(edge_complex_mask[modified_mask]) * 100.0)
            cf_stats["modified_in_selected_region_pct"] = float(np.mean(region_mask[modified_mask]) * 100.0)
            cf_stats["modified_in_cf_top_quartile_pct"] = cf_complex_pct
            cf_stats["cf_modified_complex_pct"] = cf_complex_pct
            cf_stats["cf_modified_high_complexity_pct"] = cf_complex_pct
            cf_stats["cf_modified_complex_mean"] = float(np.mean(modified_complex_values))
            cf_stats["cf_modified_complex_median"] = float(np.median(modified_complex_values))
            all_complex_mean = float(np.mean(psi_norm)) if psi_norm.size else float("nan")
            cf_stats["cf_modified_complex_lift"] = float(np.mean(modified_complex_values) / all_complex_mean) if all_complex_mean > 0 else float("nan")

            delta_tmp = stego.astype(np.int16) - cover.astype(np.int16)
            plus_tmp = delta_tmp > 0
            minus_tmp = delta_tmp < 0
            preferred_tmp = ((direction_score > 0) & plus_tmp) | ((direction_score < 0) & minus_tmp)
            decisive_modified = modified_mask & decisive_mask
            if np.count_nonzero(decisive_modified):
                pref_decisive = float(np.mean(preferred_tmp[decisive_modified]) * 100.0)
                decisive_pct = float(np.mean(decisive_mask[modified_mask]) * 100.0)
            else:
                pref_decisive = float("nan")
                decisive_pct = 0.0
            # Alias retained for downstream scripts and older summaries.
            cf_stats["modified_preferred_direction_pct"] = pref_decisive
            cf_stats["modified_preferred_direction_decisive_pct"] = pref_decisive
            cf_stats["qim_direction_decisive_modified_pct"] = decisive_pct
        else:
            cf_stats["fuzzy_edge_modified_pct"] = 0.0
            cf_stats["modified_in_selected_region_pct"] = 0.0
            cf_stats["modified_in_cf_top_quartile_pct"] = 0.0
            cf_stats["cf_modified_complex_pct"] = 0.0
            cf_stats["cf_modified_high_complexity_pct"] = 0.0
            cf_stats["cf_modified_complex_mean"] = 0.0
            cf_stats["cf_modified_complex_median"] = 0.0
            cf_stats["cf_modified_complex_lift"] = 0.0
            cf_stats["modified_preferred_direction_pct"] = 0.0
            cf_stats["modified_preferred_direction_decisive_pct"] = 0.0
            cf_stats["qim_direction_decisive_modified_pct"] = 0.0
        cf_stats["cf_complex_threshold_percentile"] = complex_thr_pct
        cf_stats["adaptive_direction_enabled"] = bool(args.qim_direction_mode == "adaptive-residual")
        # QIM-specific log fields after embedding.
        qim_stats = add_qim_modified_stats(
            qim_stats=qim_stats,
            cover=cover,
            stego=stego,
            modified_mask=modified_mask,
            rho_dit_u=rho_dit_u,
            psi_norm=psi_norm,
            direction_score=direction_score,
            high_texture_threshold=0.75,
        )

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
            "mean_cost": float(np.mean(rho_min)),
            "median_cost": float(np.median(rho_min)),
            "sum_cost": float(np.sum(rho_min)),
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
            "gamma1_path": str(args.gamma1),
            "gamma2_path": str(args.gamma2),
            "gamma1_sha256": sha256_file(Path(args.gamma1)),
            "gamma2_sha256": sha256_file(Path(args.gamma2)),
            **qim_stats,
            **base_cost_stats,
            **cf_stats,
            "seed": args.seed,
            "algo_params": json.dumps({
                "method_version": "Ours-QIM-v3.2",
                "mask": "m XOR gamma1 XOR gamma2 XOR rho_dit",
                "rho_dit": "Dither(k_stego, psi(o)) using HMAC-SHA256 over robust directional residual-energy features",
                "cost_map": "rho_ours = rho_base * g(psi_i(o), rho_dit_i)",
                "base_cost_mode": args.base_cost,
                "base_cost_map": base_cost_stats.get("base_cost_type", ""),
                "fused_suniward_weight": args.fused_suniward_weight,
                "mipod_variance_kernel": args.mipod_variance_kernel,
                "mipod_smooth_kernel": args.mipod_smooth_kernel,
                "mipod_variance_floor": args.mipod_variance_floor,
                "mipod_cost_power": args.mipod_cost_power,
                "psi": "robust_normalized_directional_residual_energy",
                "suniward_sigma": args.suniward_sigma,
                "wet_cost": args.wet_cost,
                "qim_texture_discount": args.qim_texture_discount,
                "qim_smooth_penalty": args.qim_smooth_penalty,
                "qim_dither_strength": args.qim_dither_strength,
                "qim_min_factor": args.qim_min_factor,
                "qim_max_factor": args.qim_max_factor,
                "qim_factor_normalization": not args.no_qim_factor_normalization,
                "qim_direction_mode": args.qim_direction_mode,
                "qim_direction_strength": args.qim_direction_strength,
                "qim_direction_dither_mix": args.qim_direction_dither_mix,
                "qim_direction_residual_scale": args.qim_direction_residual_scale,
                "high_variance_percentile": args.high_variance_percentile,
                "cf_features_enabled": not args.no_cf_features,
                "cf_map_type": cf_stats.get("cf_map_type", ""),
                "cf_use_slic": args.cf_use_slic,
                "slic_segments": args.slic_segments,
                "region_threshold": args.region_threshold,
                "cluster_penalty": args.cluster_penalty,
                "cost_smooth_kernel": args.cost_smooth_kernel,
                "cost_smooth_mix": args.cost_smooth_mix,
                "cf_complexity_discount": args.cf_complexity_discount,
                "cf_smooth_penalty": args.cf_smooth_penalty,
                "gamma_expansion": "file_bits_plus_hmac_expand",
                "payload_source": "hmac_deterministic" if args.deterministic_payload else "per_image_keyed_numpy_rng",
                "coding_loss_note": "cost-based proxy, not exact STC rate-distortion gap",
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
        rho_fail = locals().get("rho_min", None)
        qim_stats_fail = locals().get("qim_stats", empty_qim_stats())
        cf_stats_fail = locals().get("cf_stats", empty_cf_stats())
        base_cost_stats_fail = locals().get("base_cost_stats", {
            "base_cost_mode": getattr(args, "base_cost", ""),
            "fused_suniward_weight": getattr(args, "fused_suniward_weight", ""),
            "mipod_variance_kernel": getattr(args, "mipod_variance_kernel", ""),
            "mipod_smooth_kernel": getattr(args, "mipod_smooth_kernel", ""),
            "mipod_variance_floor": getattr(args, "mipod_variance_floor", ""),
            "mipod_cost_power": getattr(args, "mipod_cost_power", ""),
        })
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
            "gamma1_path": str(args.gamma1),
            "gamma2_path": str(args.gamma2),
            "gamma1_sha256": sha256_file(Path(args.gamma1)) if Path(args.gamma1).exists() else "",
            "gamma2_sha256": sha256_file(Path(args.gamma2)) if Path(args.gamma2).exists() else "",
            **qim_stats_fail,
            **base_cost_stats_fail,
            **cf_stats_fail,
            "seed": args.seed,
            "algo_params": "{}",
            "ok": ok,
            "error": err,
        })

    append_csv(logs_dir / "embed_log_ours_qim.csv", CSV_HEADER, row)
    append_jsonl(logs_dir / "embed_log_ours_qim.jsonl", row)
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
            "method": "Ours-QIM",
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
            "out_dir": str(out_root / f"ours_qim_{payload:.3f}"),
        }
        append_csv(logs / "embed_summary_ours_qim.csv", SUMMARY_HEADER, row)


# ------------------------------- CLI -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ours-QIM-v3 hybrid masking + conservative cover-conditioned adaptive embedding for BOSSBase.")
    p.add_argument("--boss-root", default="/home/omego/Downloads/BOSSbase", help="BOSSBase root containing .pgm images")
    p.add_argument("--out-root", default=None, help="Output root for stego images and logs")
    p.add_argument("--gamma1", default="/home/omego/Documents/Hybrid_Stego_Eva/cover_parameter/hybrid-stego/gamma1_pride_selected_seed44.txt")
    p.add_argument("--gamma2", default="/home/omego/Documents/Hybrid_Stego_Eva/cover_parameter/hybrid-stego/gamma2_pride_selected_seed44.txt")
    p.add_argument("--payloads", default="0.1,0.2,0.4", help="Comma-separated payloads in bpp")
    p.add_argument("--max-images", type=int, default=0, help="Process only first N images for testing; 0 means all")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--master-secret", default="change-this-demo-secret", help="Used only to derive deterministic evaluation bits")
    p.add_argument("--deterministic-payload", action="store_true", help="Use HMAC-derived payload bits instead of numpy RNG")
    # Legacy options retained so older command lines remain accepted. They are
    # ignored by the reference-form S-UNIWARD implementation in v2.
    p.add_argument("--wavelet", default="db8", help=argparse.SUPPRESS)
    p.add_argument("--levels", type=int, default=2, help=argparse.SUPPRESS)
    p.add_argument("--eps", type=float, default=1e-3, help=argparse.SUPPRESS)
    p.add_argument("--wavelet-mode", default="periodization", help=argparse.SUPPRESS)

    p.add_argument("--suniward-sigma", type=float, default=1.0,
                   help="S-UNIWARD stabilising constant sigma used in 1/(|residual|+sigma).")
    p.add_argument("--wet-cost", type=float, default=1e13,
                   help="Wet cost used to forbid invalid +1/-1 changes at pixel boundaries.")
    p.add_argument("--base-cost", choices=["suniward", "mipod", "fused"], default="suniward",
                   help="Base distortion model before QIM/CF modulation: suniward, mipod, or fused.")
    p.add_argument("--fused-suniward-weight", type=float, default=0.50,
                   help="Weight of S-UNIWARD in --base-cost fused; 0.5 gives equal log-domain fusion.")
    p.add_argument("--mipod-variance-kernel", type=int, default=7,
                   help="Local window size for MiPOD-like residual variance estimation.")
    p.add_argument("--mipod-smooth-kernel", type=int, default=7,
                   help="Smoothing window for MiPOD-like Fisher/cost map.")
    p.add_argument("--mipod-variance-floor", type=float, default=0.01,
                   help="Variance floor for MiPOD-like cost estimation.")
    p.add_argument("--mipod-cost-power", type=float, default=1.0,
                   help="Exponent for MiPOD-like inverse-variance cost.")
    p.add_argument("--high-variance-percentile", type=float, default=75.0)
    p.add_argument("--cf-complex-threshold-percentile", type=float, default=75.0,
                   help="Percentile threshold used for cf_modified_complex_pct; default logs top-quartile complex-feature placement.")
    p.add_argument("--qim-texture-discount", type=float, default=0.10,
                   help="Conservative discount applied to high residual-energy regions.")
    p.add_argument("--qim-smooth-penalty", type=float, default=0.10,
                   help="Conservative penalty applied to low residual-energy regions.")
    p.add_argument("--qim-dither-strength", type=float, default=0.01,
                   help="Small keyed cover-conditioned dither perturbation in the cost factor.")
    p.add_argument("--qim-min-factor", type=float, default=0.85,
                   help="Minimum multiplicative cost factor for rho_ours.")
    p.add_argument("--qim-max-factor", type=float, default=1.20,
                   help="Maximum multiplicative cost factor for rho_ours.")
    p.add_argument("--no-qim-factor-normalization", action="store_true",
                   help="Disable geometric-mean normalization of the multiplicative QIM cost factor.")
    p.add_argument("--qim-direction-mode", choices=["none", "predictor-dither", "adaptive-residual"], default="none",
                   help="Optional +1/-1 cost modulation. adaptive-residual is the V3 residual-preserving polarity mode.")
    p.add_argument("--qim-direction-strength", type=float, default=0.05,
                   help="Strength of optional directional cost modulation.")
    p.add_argument("--qim-direction-dither-mix", type=float, default=0.15,
                   help="Fraction of dither contribution in predictor-dither polarity preference.")
    p.add_argument("--qim-direction-residual-scale", type=float, default=4.0,
                   help="Residual scale used by the optional predictor-based polarity score.")
    # Ours-QIM-v3 complex-feature cost-map options. Defaults are conservative
    # so your old V2 smoke command remains a safe starting point.
    p.add_argument("--no-cf-features", action="store_true",
                   help="Disable V3 complex-feature map and use the V2 residual-energy-only psi map.")
    p.add_argument("--cf-alpha-suniward", type=float, default=0.35)
    p.add_argument("--cf-alpha-srm", type=float, default=0.25)
    p.add_argument("--cf-alpha-edge", type=float, default=0.15)
    p.add_argument("--cf-alpha-sharp", type=float, default=0.10)
    p.add_argument("--cf-alpha-var", type=float, default=0.10)
    p.add_argument("--cf-alpha-entropy", type=float, default=0.05)
    p.add_argument("--cf-sharpen-amount", type=float, default=0.15)
    p.add_argument("--cf-sharpen-sigma", type=float, default=1.0)
    p.add_argument("--cf-use-slic", action="store_true",
                   help="Enable SLIC/region-wise complex-region masking. If skimage is unavailable, block-wise fallback is used.")
    p.add_argument("--slic-segments", type=int, default=256)
    p.add_argument("--slic-compactness", type=float, default=10.0)
    p.add_argument("--slic-sigma", type=float, default=0.0)
    p.add_argument("--region-threshold", type=float, default=65.0,
                   help="Percentile threshold for selected complex regions, e.g. 65 keeps roughly top 35 percent by region score.")
    p.add_argument("--cluster-penalty", type=float, default=1.25,
                   help="Cost multiplier outside selected complex regions when --cf-use-slic is enabled.")
    p.add_argument("--cf-wet-outside-regions", action="store_true",
                   help="Make pixels outside selected complex regions wet. Strong; use only if capacity remains sufficient.")
    p.add_argument("--cost-smooth-kernel", type=int, default=3, choices=[0,1,3,5,7],
                   help="Local mean smoothing kernel for the spreading rule. 0/1 disables smoothing.")
    p.add_argument("--cost-smooth-mix", type=float, default=0.25,
                   help="Blend between original and smoothed costs for spreading rule.")
    p.add_argument("--cf-complexity-discount", type=float, default=0.08)
    p.add_argument("--cf-smooth-penalty", type=float, default=0.08)
    p.add_argument("--cf-min-factor", type=float, default=0.85)
    p.add_argument("--cf-max-factor", type=float, default=1.20)
    p.add_argument("--run-id", default=None)
    p.add_argument("--progress", choices=["auto", "bar", "none"], default="auto")
    p.add_argument("--stc-backend", choices=["auto", "pystc", "proxy"], default="pystc",
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

    gamma1_path = Path(args.gamma1)
    gamma2_path = Path(args.gamma2)
    if not boss_root.exists():
        raise FileNotFoundError(f"BOSSBase root not found: {boss_root}")
    if not gamma1_path.exists():
        raise FileNotFoundError(f"gamma1 file not found: {gamma1_path}")
    if not gamma2_path.exists():
        raise FileNotFoundError(f"gamma2 file not found: {gamma2_path}")

    run_id = args.run_id or f"ours-qim-v3-2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    payloads = [float(x.strip()) for x in args.payloads.split(",") if x.strip()]
    paths = discover_images(boss_root, max_images=args.max_images if args.max_images > 0 else None)
    if not paths:
        raise RuntimeError(f"No image files found under {boss_root}")

    gamma1_bits = load_cover_stream_bits(gamma1_path)
    gamma2_bits = load_cover_stream_bits(gamma2_path)
    master_key = args.master_secret.encode("utf-8")

    jobs = [(p, payload) for payload in payloads for p in paths]
    rows = []
    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] images={len(paths)}, payloads={payloads}, jobs={len(jobs)}")
    print(f"[INFO] gamma1_bits={gamma1_bits.size}, gamma2_bits={gamma2_bits.size}")
    print(f"[INFO] stc_backend={args.stc_backend}, safe_subprocess={args.stc_safe_subprocess}, pystc_available={pystc is not None}")

    for cover_path, payload in iter_progress(jobs, args.progress):
        row = process_one(cover_path, payload, args, run_id, gamma1_bits, gamma2_bits, master_key, logs_dir)
        rows.append(row)

    summarize(rows, out_root, run_id)
    print(f"[OK] Completed Ours-QIM run_id={run_id}")
    print(f"Logs:\n  - {logs_dir/'embed_log_ours_qim.csv'}\n  - {logs_dir/'embed_log_ours_qim.jsonl'}\n  - {logs_dir/'embed_summary_ours_qim.csv'}")


if __name__ == "__main__":
    main()
