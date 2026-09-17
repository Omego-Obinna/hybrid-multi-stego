#!/usr/bin/env python3
"""
Create camera-ready SRM+EC tables and plots from metrics.csv files.

Expected input layout:
  /home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC/runs/payload_010_seed20260605_srm_ec_fast30/metrics.csv
  /home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC/runs/payload_020_seed20260605_srm_ec_fast30/metrics.csv
  /home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC/runs/payload_040_seed20260605_srm_ec_fast30/metrics.csv

Outputs:
  paper_outputs/tables/*.csv
  paper_outputs/tables/*.tex
  paper_outputs/figures/*.pdf, *.png, *.svg

Run:
  cd /home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC
  source venv/bin/activate
  python scripts/make_srm_ec_tables_plots.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


DEFAULT_RUNS: Dict[str, str] = {
    "0.10": "/home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC/runs/payload_010_seed20260605_srm_ec_fast30",
    "0.20": "/home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC/runs/payload_020_seed20260605_srm_ec_fast30",
    "0.40": "/home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC/runs/payload_040_seed20260605_srm_ec_fast30",
}

METRIC_COLUMNS = [
    "acc", "bacc", "auc", "eer", "pe", "mcc",
    "tpr_at_fpr_0.01", "tpr_at_fpr_0.05", "tpr_at_fpr_0.10",
]


# -----------------------------
# General helpers
# -----------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_payload_float(payload_label: str) -> float:
    return float(payload_label)


def payload_code_from_label(payload_label: str) -> str:
    # "0.10" -> "010", "0.40" -> "040"
    return f"{int(round(float(payload_label) * 100)):03d}"


def clean_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def choose_split_row(df: pd.DataFrame, split: str) -> pd.Series:
    """Choose the requested split row. Falls back safely for older metrics files."""
    if "split" in df.columns:
        mask = df["split"].astype(str).str.lower().eq(split.lower())
        if mask.any():
            return df.loc[mask].iloc[0]

    # Fallback: earlier scripts used eval order val,test. If no split column exists,
    # row 1 is probably test and row 0 is probably val.
    if split.lower() == "test" and len(df) >= 2:
        return df.iloc[1]
    return df.iloc[0]


def read_metrics_for_payload(payload_label: str, run_dir: Path, split: str) -> dict:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv for payload {payload_label}: {metrics_path}")

    df = pd.read_csv(metrics_path)
    row = choose_split_row(df, split)
    config = load_json_if_exists(run_dir / "config.json")

    record = {
        "payload_bpp": payload_label,
        "payload_code": payload_code_from_label(payload_label),
        "payload_numeric": parse_payload_float(payload_label),
        "split": str(row.get("split", split)),
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_path),
    }

    for col in METRIC_COLUMNS:
        if col in row.index:
            record[col] = clean_float(row[col])
        else:
            record[col] = np.nan

    for col in ["n_pairs", "n_samples", "n_estimators_trained", "subspace_dim", "threshold", "seed"]:
        if col in row.index:
            record[col] = clean_float(row[col])
        elif col in config:
            record[col] = clean_float(config[col])
        else:
            record[col] = np.nan

    # Keep requested/trained estimators explicit when available.
    if "n_estimators_requested" in row.index:
        record["n_estimators_requested"] = clean_float(row["n_estimators_requested"])
    elif "n_estimators" in config:
        record["n_estimators_requested"] = clean_float(config["n_estimators"])
    else:
        record["n_estimators_requested"] = np.nan

    return record


def load_all_runs(runs: Dict[str, str], split: str) -> pd.DataFrame:
    rows = []
    for payload_label, run_path in runs.items():
        rows.append(read_metrics_for_payload(payload_label, Path(run_path).expanduser().resolve(), split))
    df = pd.DataFrame(rows)
    df = df.sort_values("payload_numeric").reset_index(drop=True)
    return df


# -----------------------------
# Formatting helpers
# -----------------------------

def pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{100.0 * float(x):.{digits}f}"


def dec(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def int_or_dash(x) -> str:
    if pd.isna(x):
        return "--"
    return str(int(round(float(x))))


def make_formatted_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({
        "Payload (bpp)": df["payload_bpp"],
        "Pairs": df["n_pairs"].map(int_or_dash),
        "ACC (\\%)": df["acc"].map(lambda x: pct(x, 2)),
        "BACC (\\%)": df["bacc"].map(lambda x: pct(x, 2)),
        "AUC": df["auc"].map(lambda x: dec(x, 4)),
        "EER (\\%)": df["eer"].map(lambda x: pct(x, 2)),
        "$P_e$ (\\%)": df["pe"].map(lambda x: pct(x, 2)),
        "MCC": df["mcc"].map(lambda x: dec(x, 4)),
        "TPR@FPR=0.01 (\\%)": df["tpr_at_fpr_0.01"].map(lambda x: pct(x, 2)),
        "EC": df["n_estimators_trained"].map(int_or_dash),
        "Subspace": df["subspace_dim"].map(int_or_dash),
    })
    return table


def write_markdown_table(table_df: pd.DataFrame, path: Path) -> None:
    # Avoid requiring tabulate by writing Markdown manually.
    headers = list(table_df.columns)
    rows = table_df.astype(str).values.tolist()
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")


def latex_table_string(table_df: pd.DataFrame, caption: str, label: str) -> str:
    latex = table_df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrrrrrrr",
        caption=caption,
        label=label,
    )
    note = (
        "\n% Notes: ACC = accuracy; BACC = balanced accuracy; "
        "EER = equal error rate; $P_e$ = minimum average detection error; "
        "MCC = Matthews correlation coefficient; EC = number of trained FLD ensemble members.\n"
    )
    return latex + note


# -----------------------------
# Plotting helpers
# -----------------------------

def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_auc_trend(df: pd.DataFrame, fig_dir: Path) -> None:
    x_labels = df["payload_bpp"].astype(str).tolist()
    x = np.arange(len(x_labels))
    auc = df["auc"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(x, auc, marker="o", label="SRM+EC AUC")
    ax.axhline(0.5, linestyle="--", linewidth=1.1, label="Random baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Payload (bpp)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, max(0.70, float(np.nanmax(auc)) + 0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="best")

    save_figure(fig, fig_dir / "fig_srm_ec_auc_vs_payload")


def plot_detector_metrics(df: pd.DataFrame, fig_dir: Path) -> None:
    x_labels = df["payload_bpp"].astype(str).tolist()
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(x, df["acc"].astype(float), marker="o", label="Accuracy")
    ax.plot(x, df["bacc"].astype(float), marker="s", label="Balanced accuracy")
    ax.plot(x, df["auc"].astype(float), marker="^", label="AUC")
    ax.axhline(0.5, linestyle="--", linewidth=1.1, label="Random baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Payload (bpp)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.45, 0.75)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))

    save_figure(fig, fig_dir / "fig_srm_ec_detection_scores_vs_payload")


def plot_error_metrics(df: pd.DataFrame, fig_dir: Path) -> None:
    x_labels = df["payload_bpp"].astype(str).tolist()
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(x, 100 * df["eer"].astype(float), marker="o", label="EER")
    ax.plot(x, 100 * df["pe"].astype(float), marker="s", label="$P_e$")
    ax.axhline(50.0, linestyle="--", linewidth=1.1, label="Random-error reference")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Payload (bpp)")
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(35, 55)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="best")

    save_figure(fig, fig_dir / "fig_srm_ec_error_rates_vs_payload")


def plot_tpr_low_fpr(df: pd.DataFrame, fig_dir: Path) -> None:
    x_labels = df["payload_bpp"].astype(str).tolist()
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(x, 100 * df["tpr_at_fpr_0.01"].astype(float), marker="o", label="FPR=0.01")
    ax.plot(x, 100 * df["tpr_at_fpr_0.05"].astype(float), marker="s", label="FPR=0.05")
    ax.plot(x, 100 * df["tpr_at_fpr_0.10"].astype(float), marker="^", label="FPR=0.10")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Payload (bpp)")
    ax.set_ylabel("TPR (%)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="best")

    save_figure(fig, fig_dir / "fig_srm_ec_tpr_at_low_fpr_vs_payload")


# -----------------------------
# Main program
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate camera-ready SRM+EC tables and plots from payload metrics.csv files."
    )
    ap.add_argument(
        "--out-dir",
        default="/home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC/paper_outputs",
        help="Output directory for tables and figures.",
    )
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="Which split to use for the main paper table and plots. Default: test.",
    )
    ap.add_argument(
        "--run-010",
        default=DEFAULT_RUNS["0.10"],
        help="Run directory for payload 0.10 bpp.",
    )
    ap.add_argument(
        "--run-020",
        default=DEFAULT_RUNS["0.20"],
        help="Run directory for payload 0.20 bpp.",
    )
    ap.add_argument(
        "--run-040",
        default=DEFAULT_RUNS["0.40"],
        help="Run directory for payload 0.40 bpp.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    table_dir = ensure_dir(out_dir / "tables")
    fig_dir = ensure_dir(out_dir / "figures")

    runs = {
        "0.10": args.run_010,
        "0.20": args.run_020,
        "0.40": args.run_040,
    }

    configure_matplotlib()

    # Main split table/plots, typically test.
    df_main = load_all_runs(runs, split=args.split)
    raw_csv = table_dir / f"srm_ec_{args.split}_summary_raw.csv"
    df_main.to_csv(raw_csv, index=False)

    table = make_formatted_table(df_main)
    formatted_csv = table_dir / f"srm_ec_{args.split}_summary_formatted.csv"
    formatted_tex = table_dir / f"srm_ec_{args.split}_summary.tex"
    formatted_md = table_dir / f"srm_ec_{args.split}_summary.md"

    table.to_csv(formatted_csv, index=False)
    write_markdown_table(table, formatted_md)

    caption = (
        f"SRM+EC steganalysis performance on the {args.split} split across payloads. "
        "Values are reported for Ours-QIM-v3.2-Fused-CF stego images."
    )
    label = f"tab:srm_ec_{args.split}_payloads"
    formatted_tex.write_text(latex_table_string(table, caption, label), encoding="utf-8")

    # Optional combined raw val/test summary for appendix.
    df_val = load_all_runs(runs, split="val")
    df_test = load_all_runs(runs, split="test")
    df_val_test = pd.concat([df_val, df_test], ignore_index=True)
    df_val_test.to_csv(table_dir / "srm_ec_val_test_summary_raw.csv", index=False)

    val_test_table = make_formatted_table(df_val_test)
    # Add split as first column for appendix table.
    val_test_table.insert(1, "Split", df_val_test["split"].astype(str).str.lower().tolist())
    val_test_table.to_csv(table_dir / "srm_ec_val_test_summary_formatted.csv", index=False)
    (table_dir / "srm_ec_val_test_summary.tex").write_text(
        latex_table_string(
            val_test_table,
            "SRM+EC validation and test performance across payloads.",
            "tab:srm_ec_val_test_payloads",
        ),
        encoding="utf-8",
    )

    # Figures.
    plot_auc_trend(df_main, fig_dir)
    plot_detector_metrics(df_main, fig_dir)
    plot_error_metrics(df_main, fig_dir)
    plot_tpr_low_fpr(df_main, fig_dir)

    # Plain text summary.
    summary_lines = [
        "SRM+EC paper-output generation completed.",
        f"Main split: {args.split}",
        f"Output directory: {out_dir}",
        "",
        "Main table files:",
        f"  {formatted_tex}",
        f"  {formatted_csv}",
        f"  {formatted_md}",
        "",
        "Main figures:",
        f"  {fig_dir / 'fig_srm_ec_auc_vs_payload.pdf'}",
        f"  {fig_dir / 'fig_srm_ec_detection_scores_vs_payload.pdf'}",
        f"  {fig_dir / 'fig_srm_ec_error_rates_vs_payload.pdf'}",
        f"  {fig_dir / 'fig_srm_ec_tpr_at_low_fpr_vs_payload.pdf'}",
    ]
    (out_dir / "README_outputs.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
