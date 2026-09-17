#!/usr/bin/env python3
"""
multichannel_qim_resync_sim.py

Independent discrete-event simulator for the QIM-style hybrid multichannel
steganographic protocol. The simulator evaluates operational robustness and
re-synchronisation, not steganographic detectability.

Simulated epoch layout:
    C1: H_e || (nonce_a || gamma_1)
    C2: H_e || (nonce_b || gamma_2)
    C3: H_e || (nonce_c, s_e, MAC_e)

QIM-style protocol abstraction:
    rho_dit = Dither(k_stego, psi(o))
    P'_params = gamma_1 XOR gamma_2 XOR rho_dit
    b = m XOR P'_params
    s_e = Enc(k_stego, o, b)        # abstracted as a keyed digest of b
    MAC_e = HMAC(V_pri, H_e || nonce_c || s_e)

The scheduler Sigma applies loss, jitter, skew, reordering, burst loss,
optional replay and optional tampering. The receiver buffers frames in a
sliding epoch window W, rejects stale/replayed nonces, verifies MAC on C3,
uses timeouts and heartbeat events to detect loss of synchronisation, and
optionally applies a simple erasure-recovery abstraction.

Outputs:
    run_summary.csv   one row per scenario/seed
    run_summary.jsonl same information as JSONL
    epoch_log.csv     optional detailed per-epoch log when --save-epoch-log is set
    manifest.json     reproducibility metadata

Author: generated for Hybrid Multichannel Steganography experiments
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import itertools
import json
import math
import os
import platform
import random
import statistics as stats
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CHANNELS = ("C1", "C2", "C3")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(float(x.strip())) for x in text.split(",") if x.strip()]


def percentile(values: List[float], q: float) -> float:
    """Simple percentile using linear interpolation."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def mean_or_nan(values: List[float]) -> float:
    return float(stats.mean(values)) if values else float("nan")


def median_or_nan(values: List[float]) -> float:
    return float(stats.median(values)) if values else float("nan")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hmac_digest(key: bytes, *parts: bytes, n: int = 32) -> bytes:
    h = hmac.new(key, digestmod=hashlib.sha256)
    for p in parts:
        h.update(p)
    return h.digest()[:n]


def int_to_bytes(x: int, n: int = 8) -> bytes:
    return int(x).to_bytes(n, "big", signed=False)


def xor_bytes(*chunks: bytes) -> bytes:
    if not chunks:
        return b""
    L = len(chunks[0])
    if any(len(c) != L for c in chunks):
        raise ValueError("All chunks must have equal length for XOR")
    out = bytearray(L)
    for c in chunks:
        for i, b in enumerate(c):
            out[i] ^= b
    return bytes(out)


def json_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    channel: str
    epoch: int
    header: bytes
    nonce: bytes
    payload: bytes
    mac: Optional[bytes]
    send_time_ms: float
    arrival_time_ms: float
    frame_kind: str = "data"       # data, heartbeat, replay, tampered
    tampered: bool = False
    dropped_by_scheduler: bool = False

    def frame_id(self) -> str:
        raw = b"|".join([
            self.channel.encode(),
            int_to_bytes(self.epoch),
            self.header,
            self.nonce,
            self.payload[:32],
            self.mac or b"",
            self.frame_kind.encode(),
        ])
        return sha256_bytes(raw)[:16]


@dataclass
class EpochBundle:
    epoch: int
    base_time_ms: float
    header: bytes
    gamma1: bytes
    gamma2: bytes
    rho_dit: bytes
    message: bytes
    masked_b: bytes
    stego_obj: bytes
    nonce_a: bytes
    nonce_b: bytes
    nonce_c: bytes
    mac: bytes


@dataclass
class EpochState:
    epoch: int
    first_arrival_ms: Optional[float] = None
    last_update_ms: Optional[float] = None
    c1: Optional[Frame] = None
    c2: Optional[Frame] = None
    c3: Optional[Frame] = None
    accepted: bool = False
    recovered_by_ecc: bool = False
    timed_out: bool = False
    timeout_reason: str = ""

    def present_channels(self) -> int:
        return int(self.c1 is not None) + int(self.c2 is not None) + int(self.c3 is not None)


@dataclass
class Scenario:
    loss_rate: float
    jitter_ms: float
    skew_ms: float
    window_W: int
    timeout_factor: float
    heartbeat_h: int
    ecc_enabled: bool
    burst_loss_prob: float
    burst_len: int
    replay_rate: float
    tamper_rate: float
    base_delay_ms: float
    epoch_interval_ms: float
    seed: int

    @property
    def timeout_tau_ms(self) -> float:
        return self.timeout_factor * self.skew_ms


@dataclass
class RunMetrics:
    run_id: str
    timestamp: str
    seed: int
    num_epochs: int
    payload_bytes: int
    gamma_bytes: int
    loss_rate: float
    jitter_ms: float
    skew_ms: float
    window_W: int
    timeout_tau_ms: float
    heartbeat_h: int
    ecc_enabled: bool
    burst_loss_prob: float
    burst_len: int
    replay_rate: float
    tamper_rate: float
    base_delay_ms: float
    epoch_interval_ms: float

    emitted_frames: int = 0
    delivered_frames: int = 0
    scheduler_dropped_frames: int = 0
    replay_injected_frames: int = 0
    tampered_injected_frames: int = 0

    accepted_epochs: int = 0
    dropped_epochs: int = 0
    recovered_epochs: int = 0
    resync_events: int = 0
    resync_successes: int = 0
    mac_failures: int = 0
    replay_rejections: int = 0
    stale_epoch_rejections: int = 0
    false_accepts: int = 0
    nonce_freshness_rejections: int = 0
    out_of_window_rejections: int = 0
    heartbeat_events: int = 0

    mean_resync_delay_ms: float = float("nan")
    median_resync_delay_ms: float = float("nan")
    p95_resync_delay_ms: float = float("nan")
    mean_e2e_latency_ms: float = float("nan")
    median_e2e_latency_ms: float = float("nan")
    p95_e2e_latency_ms: float = float("nan")
    throughput_epochs_per_sec: float = 0.0
    throughput_bits_per_sec: float = 0.0
    integrity_failure_rate: float = 0.0
    acceptance_rate: float = 0.0
    recovery_rate: float = 0.0
    resync_success_rate: float = 0.0
    simulated_duration_ms: float = 0.0
    wall_time_sec: float = 0.0


# ---------------------------------------------------------------------------
# Protocol simulator: sender and scheduler
# ---------------------------------------------------------------------------


class QIMHybridSender:
    def __init__(self, session_key: bytes, v_pri: bytes, payload_bytes: int, gamma_bytes: int):
        if payload_bytes != gamma_bytes:
            # Keeping equal lengths makes the XOR masking explicit and clean.
            raise ValueError("payload_bytes and gamma_bytes should be equal in this simulator")
        self.session_key = session_key
        self.v_pri = v_pri
        self.payload_bytes = payload_bytes
        self.gamma_bytes = gamma_bytes
        self.k_stego = hmac_digest(session_key, b"k_stego", n=32)
        self.k_header = hmac_digest(session_key, b"k_header", n=32)
        self.k_nonce = hmac_digest(session_key, b"k_nonce", n=32)
        self.k_message = hmac_digest(session_key, b"k_message", n=32)

    def make_epoch(self, e: int, base_time_ms: float) -> EpochBundle:
        eb = int_to_bytes(e)
        # H_e is content-independent: derived only from session key and epoch.
        header = hmac_digest(self.k_header, b"H_e", eb, n=16)
        nonce_a = hmac_digest(self.k_nonce, b"nonce_a", eb, n=16)
        nonce_b = hmac_digest(self.k_nonce, b"nonce_b", eb, n=16)
        nonce_c = hmac_digest(self.k_nonce, b"nonce_c", eb, n=16)

        gamma1 = hmac_digest(self.session_key, b"gamma1", eb, n=self.gamma_bytes)
        gamma2 = hmac_digest(self.session_key, b"gamma2", eb, n=self.gamma_bytes)

        # psi(o) is a deterministic cover-feature abstraction for the epoch.
        # rho_dit = Dither(k_stego, psi(o)).
        psi_o = hmac_digest(self.session_key, b"psi_cover_features", eb, n=32)
        rho_dit = hmac_digest(self.k_stego, b"rho_dit", psi_o, n=self.payload_bytes)

        m = hmac_digest(self.k_message, b"message", eb, n=self.payload_bytes)
        masked_b = xor_bytes(m, gamma1, gamma2, rho_dit)

        # Abstract stego object: deterministic keyed representation of Enc(k, o, b).
        stego_obj = hmac_digest(self.k_stego, b"stego_object", eb, masked_b, n=self.payload_bytes)
        mac = hmac_digest(self.v_pri, header, nonce_c, stego_obj, n=32)

        return EpochBundle(
            epoch=e, base_time_ms=base_time_ms, header=header,
            gamma1=gamma1, gamma2=gamma2, rho_dit=rho_dit,
            message=m, masked_b=masked_b, stego_obj=stego_obj,
            nonce_a=nonce_a, nonce_b=nonce_b, nonce_c=nonce_c, mac=mac,
        )

    def emit_frames(self, bundle: EpochBundle) -> List[Frame]:
        t = bundle.base_time_ms
        return [
            Frame("C1", bundle.epoch, bundle.header, bundle.nonce_a, bundle.gamma1, None, t, t),
            Frame("C2", bundle.epoch, bundle.header, bundle.nonce_b, bundle.gamma2, None, t, t),
            Frame("C3", bundle.epoch, bundle.header, bundle.nonce_c, bundle.stego_obj, bundle.mac, t, t),
        ]


class BurstLossModel:
    def __init__(self, rng: random.Random, p_enter: float, burst_len: int):
        self.rng = rng
        self.p_enter = float(max(0.0, p_enter))
        self.burst_len = max(0, int(burst_len))
        self.remaining = {ch: 0 for ch in CHANNELS}

    def drop(self, channel: str) -> bool:
        if self.burst_len <= 0 or self.p_enter <= 0.0:
            return False
        if self.remaining[channel] > 0:
            self.remaining[channel] -= 1
            return True
        if self.rng.random() < self.p_enter:
            self.remaining[channel] = self.burst_len - 1
            return True
        return False


class SchedulerSigma:
    def __init__(self, scenario: Scenario, rng: random.Random):
        self.scenario = scenario
        self.rng = rng
        self.burst = BurstLossModel(rng, scenario.burst_loss_prob, scenario.burst_len)

    def channel_skew(self, channel: str) -> float:
        S = self.scenario.skew_ms
        if channel == "C1":
            return self.rng.uniform(-S, S)
        if channel == "C2":
            return self.rng.uniform(-S, S)
        return self.rng.uniform(-S, S)

    def jitter(self) -> float:
        J = self.scenario.jitter_ms
        if J <= 0:
            return 0.0
        # Truncated Gaussian jitter, in milliseconds.
        x = self.rng.gauss(0.0, J / 2.0)
        return max(-J, min(J, x))

    def transmit(self, frames: List[Frame]) -> Tuple[List[Frame], Dict[str, int]]:
        delivered: List[Frame] = []
        counts = {"dropped": 0, "replay": 0, "tampered": 0}

        for f in frames:
            # Loss due to independent packet loss or burst loss.
            if self.rng.random() < self.scenario.loss_rate or self.burst.drop(f.channel):
                f.dropped_by_scheduler = True
                counts["dropped"] += 1
                continue

            delay = self.scenario.base_delay_ms + self.channel_skew(f.channel) + self.jitter()
            delay = max(0.0, delay)
            f.arrival_time_ms = f.send_time_ms + delay
            delivered.append(f)

            # Optional replay injection: duplicate a valid frame with later arrival.
            if self.scenario.replay_rate > 0 and self.rng.random() < self.scenario.replay_rate:
                replay = Frame(
                    channel=f.channel, epoch=f.epoch, header=f.header, nonce=f.nonce,
                    payload=f.payload, mac=f.mac, send_time_ms=f.send_time_ms,
                    arrival_time_ms=f.arrival_time_ms + self.rng.uniform(1.0, 3.0 * max(1.0, self.scenario.timeout_tau_ms)),
                    frame_kind="replay",
                )
                delivered.append(replay)
                counts["replay"] += 1

            # Optional tampering injection: modified payload/MAC should be rejected.
            if self.scenario.tamper_rate > 0 and self.rng.random() < self.scenario.tamper_rate:
                bad_payload = bytearray(f.payload)
                if bad_payload:
                    bad_payload[0] ^= 0x01
                bad_mac = f.mac
                if f.channel == "C3" and bad_mac is not None:
                    bad_mac = bytearray(bad_mac)
                    bad_mac[0] ^= 0x01
                    bad_mac = bytes(bad_mac)
                tampered = Frame(
                    channel=f.channel, epoch=f.epoch, header=f.header, nonce=f.nonce,
                    payload=bytes(bad_payload), mac=bad_mac, send_time_ms=f.send_time_ms,
                    arrival_time_ms=f.arrival_time_ms + self.rng.uniform(0.0, max(1.0, self.scenario.timeout_tau_ms)),
                    frame_kind="tampered", tampered=True,
                )
                delivered.append(tampered)
                counts["tampered"] += 1

        return delivered, counts


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------


class Receiver:
    def __init__(self, v_pri: bytes, scenario: Scenario, num_epochs: int, payload_bits_per_epoch: int):
        self.v_pri = v_pri
        self.scenario = scenario
        self.num_epochs = num_epochs
        self.payload_bits_per_epoch = payload_bits_per_epoch
        self.buffers: Dict[int, EpochState] = {}
        self.seen_nonces: set[bytes] = set()
        self.accepted_epochs: Dict[int, float] = {}
        self.dropped_epochs: Dict[int, str] = {}
        self.recovered_epochs: set[int] = set()
        self.false_accepts = 0
        self.mac_failures = 0
        self.replay_rejections = 0
        self.stale_epoch_rejections = 0
        self.nonce_freshness_rejections = 0
        self.out_of_window_rejections = 0
        self.resync_events = 0
        self.resync_successes = 0
        self.heartbeat_events = 0
        self.latencies_ms: List[float] = []
        self.resync_delays_ms: List[float] = []
        self.last_accepted_epoch = -1
        self.in_sync_loss = False
        self.sync_loss_start_ms: Optional[float] = None
        self.epoch_log_rows: List[Dict[str, object]] = []

    def expected_base_time(self, epoch: int) -> float:
        return epoch * self.scenario.epoch_interval_ms

    def get_state(self, epoch: int) -> EpochState:
        if epoch not in self.buffers:
            self.buffers[epoch] = EpochState(epoch=epoch)
        return self.buffers[epoch]

    def verify_c3_mac(self, f: Frame) -> bool:
        if f.mac is None:
            return False
        expected = hmac_digest(self.v_pri, f.header, f.nonce, f.payload, n=32)
        return hmac.compare_digest(expected, f.mac)

    def frame_is_stale_or_out_of_window(self, f: Frame) -> bool:
        # Reject frames too far behind the highest accepted epoch.
        if f.epoch < self.last_accepted_epoch - self.scenario.window_W:
            self.stale_epoch_rejections += 1
            return True
        # Reject frames too far ahead of current synchronisation frontier.
        if self.last_accepted_epoch >= 0 and f.epoch > self.last_accepted_epoch + self.scenario.window_W + 1:
            self.out_of_window_rejections += 1
            return True
        return False

    def process_frame(self, f: Frame):
        now = f.arrival_time_ms
        if f.frame_kind == "heartbeat":
            self.heartbeat_events += 1
            self.flush_timeouts(now)
            return

        if self.frame_is_stale_or_out_of_window(f):
            return

        # Freshness/replay rejection: same nonce cannot be accepted twice.
        if f.nonce in self.seen_nonces:
            if f.frame_kind == "replay":
                self.replay_rejections += 1
            else:
                self.nonce_freshness_rejections += 1
            return

        # MAC verification on C3 happens before buffering C3.
        if f.channel == "C3" and not self.verify_c3_mac(f):
            self.mac_failures += 1
            return

        st = self.get_state(f.epoch)
        if st.first_arrival_ms is None:
            st.first_arrival_ms = now
        st.last_update_ms = now

        if f.channel == "C1":
            st.c1 = f
        elif f.channel == "C2":
            st.c2 = f
        elif f.channel == "C3":
            st.c3 = f
        else:
            raise ValueError(f"Unknown channel {f.channel}")

        self.try_accept(st, now)
        self.flush_timeouts(now)

    def try_accept(self, st: EpochState, now: float):
        if st.accepted or st.timed_out:
            return

        full = st.c1 is not None and st.c2 is not None and st.c3 is not None
        recovered = False
        if not full and self.scenario.ecc_enabled:
            # Simple erasure-recovery abstraction: with a C3 frame and any one of the
            # two parameter frames, a parity/checkpoint code can reconstruct the missing
            # cover parameter for this epoch. This is intentionally conservative: C3 is
            # still required because MAC binds the payload stego object.
            recovered = st.c3 is not None and ((st.c1 is not None) ^ (st.c2 is not None))

        if not (full or recovered):
            return

        # Header consistency check. A false accept would be a corrupted/tampered frame
        # accepted as valid; this should remain zero under MAC+freshness.
        headers = [x.header for x in (st.c1, st.c2, st.c3) if x is not None]
        if len(set(headers)) != 1:
            self.false_accepts += 1
            return

        # Mark nonces as seen only after successful epoch acceptance. This allows
        # duplicate arrivals before completion to be handled by overwriting the buffer,
        # but once accepted, replays are rejected.
        for f in (st.c1, st.c2, st.c3):
            if f is not None:
                self.seen_nonces.add(f.nonce)

        st.accepted = True
        st.recovered_by_ecc = recovered
        self.accepted_epochs[st.epoch] = now
        if recovered:
            self.recovered_epochs.add(st.epoch)

        latency = now - self.expected_base_time(st.epoch)
        self.latencies_ms.append(latency)

        # Resynchronisation: if at least one previous epoch was dropped/timed out,
        # the next accepted epoch marks a successful re-sync.
        if self.in_sync_loss:
            self.resync_successes += 1
            if self.sync_loss_start_ms is not None:
                self.resync_delays_ms.append(max(0.0, now - self.sync_loss_start_ms))
            self.in_sync_loss = False
            self.sync_loss_start_ms = None

        # If we jump beyond a gap, count a re-sync event.
        if self.last_accepted_epoch >= 0 and st.epoch > self.last_accepted_epoch + 1:
            self.resync_events += 1
        self.last_accepted_epoch = max(self.last_accepted_epoch, st.epoch)

        self.epoch_log_rows.append({
            "epoch": st.epoch,
            "status": "accepted_recovered" if recovered else "accepted",
            "accept_time_ms": now,
            "latency_ms": latency,
            "present_channels": st.present_channels(),
            "reason": "ecc_recovery" if recovered else "complete_epoch",
        })

    def flush_timeouts(self, now: float):
        tau = self.scenario.timeout_tau_ms
        if tau <= 0:
            return
        for e, st in list(self.buffers.items()):
            if st.accepted or st.timed_out:
                continue
            if st.first_arrival_ms is not None and now - st.first_arrival_ms > tau:
                st.timed_out = True
                st.timeout_reason = "timeout_incomplete_epoch"
                self.dropped_epochs[e] = st.timeout_reason
                if not self.in_sync_loss:
                    self.in_sync_loss = True
                    self.sync_loss_start_ms = now
                    self.resync_events += 1
                self.epoch_log_rows.append({
                    "epoch": e,
                    "status": "dropped_timeout",
                    "accept_time_ms": "",
                    "latency_ms": "",
                    "present_channels": st.present_channels(),
                    "reason": st.timeout_reason,
                })

    def finalize(self, final_time_ms: float):
        # Flush any remaining incomplete states at the end.
        self.flush_timeouts(final_time_ms + max(1.0, self.scenario.timeout_tau_ms) + 1.0)

    def build_metrics(self, run_id: str, timestamp: str, scenario: Scenario,
                      num_epochs: int, payload_bytes: int, gamma_bytes: int,
                      emitted_frames: int, delivered_frames: int, scheduler_dropped_frames: int,
                      replay_injected: int, tampered_injected: int, wall_time_sec: float,
                      simulated_duration_ms: float) -> RunMetrics:
        m = RunMetrics(
            run_id=run_id,
            timestamp=timestamp,
            seed=scenario.seed,
            num_epochs=num_epochs,
            payload_bytes=payload_bytes,
            gamma_bytes=gamma_bytes,
            loss_rate=scenario.loss_rate,
            jitter_ms=scenario.jitter_ms,
            skew_ms=scenario.skew_ms,
            window_W=scenario.window_W,
            timeout_tau_ms=scenario.timeout_tau_ms,
            heartbeat_h=scenario.heartbeat_h,
            ecc_enabled=scenario.ecc_enabled,
            burst_loss_prob=scenario.burst_loss_prob,
            burst_len=scenario.burst_len,
            replay_rate=scenario.replay_rate,
            tamper_rate=scenario.tamper_rate,
            base_delay_ms=scenario.base_delay_ms,
            epoch_interval_ms=scenario.epoch_interval_ms,
        )
        m.emitted_frames = emitted_frames
        m.delivered_frames = delivered_frames
        m.scheduler_dropped_frames = scheduler_dropped_frames
        m.replay_injected_frames = replay_injected
        m.tampered_injected_frames = tampered_injected
        m.accepted_epochs = len(self.accepted_epochs)
        m.dropped_epochs = len(self.dropped_epochs)
        m.recovered_epochs = len(self.recovered_epochs)
        m.resync_events = self.resync_events
        m.resync_successes = self.resync_successes
        m.mac_failures = self.mac_failures
        m.replay_rejections = self.replay_rejections
        m.stale_epoch_rejections = self.stale_epoch_rejections
        m.nonce_freshness_rejections = self.nonce_freshness_rejections
        m.out_of_window_rejections = self.out_of_window_rejections
        m.false_accepts = self.false_accepts
        m.heartbeat_events = self.heartbeat_events
        m.mean_resync_delay_ms = mean_or_nan(self.resync_delays_ms)
        m.median_resync_delay_ms = median_or_nan(self.resync_delays_ms)
        m.p95_resync_delay_ms = percentile(self.resync_delays_ms, 95)
        m.mean_e2e_latency_ms = mean_or_nan(self.latencies_ms)
        m.median_e2e_latency_ms = median_or_nan(self.latencies_ms)
        m.p95_e2e_latency_ms = percentile(self.latencies_ms, 95)
        m.simulated_duration_ms = simulated_duration_ms
        duration_sec = max(1e-9, simulated_duration_ms / 1000.0)
        m.throughput_epochs_per_sec = m.accepted_epochs / duration_sec
        m.throughput_bits_per_sec = (m.accepted_epochs * payload_bytes * 8) / duration_sec
        m.integrity_failure_rate = self.false_accepts / max(1, num_epochs)
        m.acceptance_rate = m.accepted_epochs / max(1, num_epochs)
        m.recovery_rate = m.recovered_epochs / max(1, num_epochs)
        m.resync_success_rate = m.resync_successes / max(1, m.resync_events)
        m.wall_time_sec = wall_time_sec
        return m


# ---------------------------------------------------------------------------
# Running one scenario and factorial designs
# ---------------------------------------------------------------------------


def make_heartbeat_events(num_epochs: int, heartbeat_h: int, epoch_interval_ms: float, base_delay_ms: float) -> List[Frame]:
    events: List[Frame] = []
    if heartbeat_h <= 0:
        return events
    for e in range(heartbeat_h - 1, num_epochs, heartbeat_h):
        t = e * epoch_interval_ms + base_delay_ms + 0.1
        for ch in CHANNELS:
            events.append(Frame(
                channel=ch,
                epoch=e,
                header=b"HEARTBEAT",
                nonce=f"hb-{ch}-{e}".encode(),
                payload=b"",
                mac=None,
                send_time_ms=t,
                arrival_time_ms=t,
                frame_kind="heartbeat",
            ))
    return events


def run_one_scenario(args, scenario: Scenario, out_dir: Path) -> Tuple[RunMetrics, List[Dict[str, object]]]:
    t0 = time.time()
    rng = random.Random(scenario.seed)
    run_id = f"qim-resync-{now_utc_iso().replace(':','').replace('-','')}-{uuid.uuid4().hex[:8]}"
    timestamp = now_utc_iso()

    session_key = hmac_digest(args.master_secret.encode(), b"session", int_to_bytes(scenario.seed), n=32)
    v_pri = hmac_digest(args.master_secret.encode(), b"V_pri", int_to_bytes(scenario.seed), n=32)
    sender = QIMHybridSender(session_key, v_pri, args.payload_bytes, args.gamma_bytes)
    scheduler = SchedulerSigma(scenario, rng)
    receiver = Receiver(v_pri, scenario, args.num_epochs, args.payload_bytes * 8)

    all_events: List[Frame] = []
    emitted_frames = 0
    delivered_frames = 0
    scheduler_dropped = 0
    replay_injected = 0
    tampered_injected = 0

    for e in range(args.num_epochs):
        base_t = e * scenario.epoch_interval_ms
        bundle = sender.make_epoch(e, base_t)
        frames = sender.emit_frames(bundle)
        emitted_frames += len(frames)
        delivered, counts = scheduler.transmit(frames)
        all_events.extend(delivered)
        delivered_frames += len(delivered)
        scheduler_dropped += counts["dropped"]
        replay_injected += counts["replay"]
        tampered_injected += counts["tampered"]

    all_events.extend(make_heartbeat_events(args.num_epochs, scenario.heartbeat_h,
                                             scenario.epoch_interval_ms, scenario.base_delay_ms))
    all_events.sort(key=lambda f: (f.arrival_time_ms, f.epoch, f.channel))

    for ev in all_events:
        receiver.process_frame(ev)

    final_time_ms = max([f.arrival_time_ms for f in all_events], default=args.num_epochs * scenario.epoch_interval_ms)
    receiver.finalize(final_time_ms)
    wall = time.time() - t0

    metrics = receiver.build_metrics(
        run_id, timestamp, scenario, args.num_epochs, args.payload_bytes, args.gamma_bytes,
        emitted_frames, delivered_frames, scheduler_dropped, replay_injected, tampered_injected,
        wall, final_time_ms,
    )
    return metrics, receiver.epoch_log_rows


def scenario_product(args) -> Iterable[Scenario]:
    seeds = parse_csv_ints(args.seeds)
    for (loss, jitter, skew, timeout_factor, W, h, ecc, burst_p, seed) in itertools.product(
        parse_csv_floats(args.loss_rates),
        parse_csv_floats(args.jitter_ms),
        parse_csv_floats(args.skew_ms),
        parse_csv_floats(args.timeout_factors),
        parse_csv_ints(args.window_W),
        parse_csv_ints(args.heartbeat_h),
        [x.strip().lower() in ("1", "true", "yes", "on") for x in args.ecc_enabled.split(",") if x.strip()],
        parse_csv_floats(args.burst_loss_probs),
        seeds,
    ):
        yield Scenario(
            loss_rate=loss,
            jitter_ms=jitter,
            skew_ms=skew,
            timeout_factor=timeout_factor,
            window_W=W,
            heartbeat_h=h,
            ecc_enabled=ecc,
            burst_loss_prob=burst_p,
            burst_len=args.burst_len,
            replay_rate=args.replay_rate,
            tamper_rate=args.tamper_rate,
            base_delay_ms=args.base_delay_ms,
            epoch_interval_ms=args.epoch_interval_ms,
            seed=seed,
        )


def write_csv_row(path: Path, row: Dict[str, object]):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_jsonl_row(path: Path, row: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def write_epoch_log(path: Path, run_id: str, scenario: Scenario, rows: List[Dict[str, object]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "run_id", "seed", "loss_rate", "jitter_ms", "skew_ms", "window_W",
        "timeout_tau_ms", "heartbeat_h", "ecc_enabled", "epoch", "status",
        "accept_time_ms", "latency_ms", "present_channels", "reason",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for r in rows:
            row = {
                "run_id": run_id,
                "seed": scenario.seed,
                "loss_rate": scenario.loss_rate,
                "jitter_ms": scenario.jitter_ms,
                "skew_ms": scenario.skew_ms,
                "window_W": scenario.window_W,
                "timeout_tau_ms": scenario.timeout_tau_ms,
                "heartbeat_h": scenario.heartbeat_h,
                "ecc_enabled": scenario.ecc_enabled,
                **r,
            }
            writer.writerow(row)


def clean_float_for_csv(row: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            if math.isnan(v):
                out[k] = ""
            else:
                out[k] = f"{v:.6f}"
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Independent QIM hybrid multichannel robustness and re-synchronisation simulator."
    )
    p.add_argument("--out-dir", required=True, help="Output directory for summary CSV/JSONL logs.")
    p.add_argument("--num-epochs", type=int, default=5000, help="Epochs per scenario/seed.")
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated seeds.")
    p.add_argument("--master-secret", default="qim-hybrid-sim-secret", help="Master secret used for deterministic HMAC derivations.")
    p.add_argument("--payload-bytes", type=int, default=32, help="Payload bytes per epoch; equals gamma bytes in this simulator.")
    p.add_argument("--gamma-bytes", type=int, default=32, help="Gamma bytes per epoch; should equal payload bytes.")

    p.add_argument("--loss-rates", default="0,0.01,0.05", help="Comma-separated independent frame loss rates.")
    p.add_argument("--jitter-ms", default="0,25,50,100", help="Comma-separated jitter magnitudes in ms.")
    p.add_argument("--skew-ms", default="50,100,200", help="Comma-separated skew bounds Delta in ms.")
    p.add_argument("--timeout-factors", default="2,3", help="Timeout tau is factor * skew_ms.")
    p.add_argument("--window-W", default="3,5,10", help="Comma-separated sliding window sizes in epochs.")
    p.add_argument("--heartbeat-h", default="5,10,20", help="Comma-separated heartbeat cadences in epochs.")
    p.add_argument("--ecc-enabled", default="false,true", help="Comma-separated booleans: false,true.")
    p.add_argument("--burst-loss-probs", default="0", help="Comma-separated probabilities to enter burst loss.")
    p.add_argument("--burst-len", type=int, default=3, help="Burst length in frames per channel.")

    p.add_argument("--replay-rate", type=float, default=0.0, help="Optional replay injection probability per delivered frame.")
    p.add_argument("--tamper-rate", type=float, default=0.0, help="Optional tamper injection probability per delivered frame.")
    p.add_argument("--base-delay-ms", type=float, default=20.0, help="Base per-frame delay in ms before skew/jitter.")
    p.add_argument("--epoch-interval-ms", type=float, default=100.0, help="Epoch emission interval in ms.")

    p.add_argument("--max-scenarios", type=int, default=0, help="Limit number of scenario/seed runs for debugging; 0 means all.")
    p.add_argument("--save-epoch-log", action="store_true", help="Write detailed per-epoch log. Can be large.")
    p.add_argument("--print-every", type=int, default=1, help="Print progress every N scenarios.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": now_utc_iso(),
        "script": Path(__file__).name,
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join([sys.executable] + sys.argv),
        "args": vars(args),
        "notes": "Protocol-level robustness/re-synchronisation simulation; not stego detectability.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    scenarios = list(scenario_product(args))
    if args.max_scenarios and args.max_scenarios > 0:
        scenarios = scenarios[:args.max_scenarios]

    print(f"[INFO] scenarios={len(scenarios)}, num_epochs={args.num_epochs}")
    print(f"[INFO] output={out_dir}")

    summary_csv = out_dir / "run_summary.csv"
    summary_jsonl = out_dir / "run_summary.jsonl"
    epoch_csv = out_dir / "epoch_log.csv"

    for idx, sc in enumerate(scenarios, start=1):
        metrics, epoch_rows = run_one_scenario(args, sc, out_dir)
        row = clean_float_for_csv(asdict(metrics))
        write_csv_row(summary_csv, row)
        write_jsonl_row(summary_jsonl, asdict(metrics))
        if args.save_epoch_log:
            write_epoch_log(epoch_csv, metrics.run_id, sc, epoch_rows)

        if idx == 1 or idx % max(1, args.print_every) == 0 or idx == len(scenarios):
            print(
                f"[RUN {idx}/{len(scenarios)}] seed={sc.seed} loss={sc.loss_rate} "
                f"jitter={sc.jitter_ms} skew={sc.skew_ms} W={sc.window_W} "
                f"tau={sc.timeout_tau_ms:.1f} h={sc.heartbeat_h} ecc={sc.ecc_enabled} "
                f"accepted={metrics.accepted_epochs}/{args.num_epochs} "
                f"resync_success_rate={metrics.resync_success_rate:.3f} "
                f"false_accepts={metrics.false_accepts} wall={metrics.wall_time_sec:.2f}s"
            )

    print(f"[OK] Summary CSV:  {summary_csv}")
    print(f"[OK] Summary JSONL: {summary_jsonl}")
    if args.save_epoch_log:
        print(f"[OK] Epoch log:    {epoch_csv}")
    print(f"[OK] Manifest:     {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
