#!/usr/bin/env python3
"""
Create camera-ready SRM+EC tables and more visually engaging plots from metrics.csv files.

This version avoids plain line plots and instead produces:
  1) AUC lollipop / margin-from-random plot
  2) Grouped performance bars for ACC, BACC, and AUC
  3) Grouped error-rate bars for EER and P_e
  4) Grouped low-FPR TPR bars
  5) Compact metric heatmap across payloads

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
  python scripts/make_srm_ec_tables_plots_v2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_payload_float(payload_label: str) -> float:
    return float(payload_label)


def payload_code_from_label(payload_label: str) -> str:
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
        record[col] = clean_float(row[col]) if col in row.index else np.nan

    for col in ["n_pairs", "n_samples", "n_estimators_trained", "subspace_dim", "threshold", "seed"]:
        if col in row.index:
            record[col] = clean_float(row[col])
        elif col in config:
            record[col] = clean_float(config[col])
        else:
            record[col] = np.nan

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
    return df.sort_values("payload_numeric").reset_index(drop=True)


# ---------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------

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
    return pd.DataFrame({
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


def write_markdown_table(table_df: pd.DataFrame, path: Path) -> None:
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


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 600,
        #"font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.45,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def payload_axis(df: pd.DataFrame):
    labels = df["payload_bpp"].astype(str).tolist()
    x = np.arange(len(labels))
    return x, labels


def annotate_values(ax, xs, ys, fmt="{:.3f}", dy=0.012, fontsize=8):
    for x, y in zip(xs, ys):
        if pd.isna(y):
            continue
        ax.text(x, y + dy, fmt.format(float(y)), ha="center", va="bottom", fontsize=fontsize)


def plot_auc_lollipop(df: pd.DataFrame, fig_dir: Path) -> None:
    """Lollipop plot: AUC relative to the random baseline."""
    x, x_labels = payload_axis(df)
    auc = df["auc"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(4.9, 3.25))
    ax.axhspan(0.49, 0.51, alpha=0.12, label="Near-random band")
    ax.axhline(0.5, linestyle="--", linewidth=1.1, label="Random baseline")

    # Lollipop stems from 0.5 baseline to observed AUC.
    for xi, yi in zip(x, auc):
        ax.vlines(xi, 0.5, yi, linewidth=2.2, alpha=0.85)
    ax.scatter(x, auc, s=78, zorder=3, edgecolor="black", linewidth=0.7, label="Test AUC")

    annotate_values(ax, x, auc, fmt="{:.3f}", dy=0.010)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Payload (bpp)")
    ax.set_ylabel("AUC")
    ymin = min(0.47, float(np.nanmin(auc)) - 0.025)
    ymax = max(0.66, float(np.nanmax(auc)) + 0.055)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, axis="y", alpha=0.32)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=1)

    save_figure(fig, fig_dir / "fig_srm_ec_auc_lollipop_vs_payload")


def plot_score_grouped_bars(df: pd.DataFrame, fig_dir: Path) -> None:
    """Grouped bars: ACC, BACC, and AUC for each payload."""
    x, x_labels = payload_axis(df)
    metrics = ["acc", "bacc", "auc"]
    labels = ["ACC", "BACC", "AUC"]
    width = 0.23
    offsets = np.array([-width, 0.0, width])

    fig, ax = plt.subplots(figsize=(5.6, 3.35))
    ax.axhspan(0.49, 0.51, alpha=0.10)
    ax.axhline(0.5, linestyle="--", linewidth=1.0, label="Random baseline")

    for metric, label, off in zip(metrics, labels, offsets):
        values = df[metric].astype(float).to_numpy()
        bars = ax.bar(x + off, values, width=width, label=label, alpha=0.92)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Payload (bpp)")
    ax.set_ylabel("Detection Score")
    ax.set_ylim(0.45, 0.72)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, axis="y", alpha=0.30)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    save_figure(fig, fig_dir / "fig_srm_ec_grouped_scores_vs_payload")


def plot_error_grouped_bars(df: pd.DataFrame, fig_dir: Path) -> None:
    """Grouped bars: EER and P_e error rates."""
    x, x_labels = payload_axis(df)
    width = 0.30
    eer = 100.0 * df["eer"].astype(float).to_numpy()
    pe = 100.0 * df["pe"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(5.2, 3.25))
    ax.axhline(50.0, linestyle="--", linewidth=1.0, label="Random-error reference")
    bars1 = ax.bar(x - width / 2, eer, width=width, label="EER", alpha=0.92)
    bars2 = ax.bar(x + width / 2, pe, width=width, label="$P_e$", alpha=0.92)

    ax.bar_label(bars1, labels=[f"{v:.1f}" for v in eer], padding=2, fontsize=8)
    ax.bar_label(bars2, labels=[f"{v:.1f}" for v in pe], padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Payload (bpp)")
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(35, 55)
    ax.grid(True, axis="y", alpha=0.30)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    save_figure(fig, fig_dir / "fig_srm_ec_grouped_error_rates_vs_payload")


def plot_tpr_low_fpr_grouped_bars(df: pd.DataFrame, fig_dir: Path) -> None:
    """Grouped bars: TPR at low false-positive rates."""
    x, x_labels = payload_axis(df)
    metrics = ["tpr_at_fpr_0.01", "tpr_at_fpr_0.05", "tpr_at_fpr_0.10"]
    labels = ["FPR=0.01", "FPR=0.05", "FPR=0.10"]
    width = 0.23
    offsets = np.array([-width, 0.0, width])

    fig, ax = plt.subplots(figsize=(5.6, 3.35))
    for metric, label, off in zip(metrics, labels, offsets):
        values = 100.0 * df[metric].astype(float).to_numpy()
        bars = ax.bar(x + off, values, width=width, label=label, alpha=0.92)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Payload (bpp)")
    ax.set_ylabel("TPR (%)")
    ymax = max(12.0, float(np.nanmax([100.0 * df[m].astype(float).max() for m in metrics])) + 4.0)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.30)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    save_figure(fig, fig_dir / "fig_srm_ec_grouped_tpr_low_fpr_vs_payload")


def plot_metric_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """Compact heatmap for the most important test metrics."""
    metrics = ["acc", "bacc", "auc", "eer", "pe", "mcc"]
    metric_labels = ["ACC", "BACC", "AUC", "EER", "$P_e$", "MCC"]
    payload_labels = df["payload_bpp"].astype(str).tolist()

    matrix = df[metrics].astype(float).to_numpy().T

    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(payload_labels)))
    ax.set_xticklabels(payload_labels)
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Payload (bpp)")

    # Write values inside cells.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:.3f}" if metric_labels[i] != "MCC" else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.ax.set_ylabel("Metric value", rotation=270, labelpad=12)

    ax.set_title("SRM+EC metric matrix")
    save_figure(fig, fig_dir / "fig_srm_ec_metric_heatmap")


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate camera-ready SRM+EC tables and polished non-line plots from payload metrics.csv files."
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
    ap.add_argument("--run-010", default=DEFAULT_RUNS["0.10"], help="Run directory for payload 0.10 bpp.")
    ap.add_argument("--run-020", default=DEFAULT_RUNS["0.20"], help="Run directory for payload 0.20 bpp.")
    ap.add_argument("--run-040", default=DEFAULT_RUNS["0.40"], help="Run directory for payload 0.40 bpp.")
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

    # Optional validation/test table for appendix.
    df_val = load_all_runs(runs, split="val")
    df_test = load_all_runs(runs, split="test")
    df_val_test = pd.concat([df_val, df_test], ignore_index=True)
    df_val_test.to_csv(table_dir / "srm_ec_val_test_summary_raw.csv", index=False)

    val_test_table = make_formatted_table(df_val_test)
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

    # New non-line figures.
    plot_auc_lollipop(df_main, fig_dir)
    plot_score_grouped_bars(df_main, fig_dir)
    plot_error_grouped_bars(df_main, fig_dir)
    plot_tpr_low_fpr_grouped_bars(df_main, fig_dir)
    plot_metric_heatmap(df_main, fig_dir)

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
        "New camera-ready figures:",
        f"  {fig_dir / 'fig_srm_ec_auc_lollipop_vs_payload.pdf'}",
        f"  {fig_dir / 'fig_srm_ec_grouped_scores_vs_payload.pdf'}",
        f"  {fig_dir / 'fig_srm_ec_grouped_error_rates_vs_payload.pdf'}",
        f"  {fig_dir / 'fig_srm_ec_grouped_tpr_low_fpr_vs_payload.pdf'}",
        f"  {fig_dir / 'fig_srm_ec_metric_heatmap.pdf'}",
    ]
    (out_dir / "README_outputs.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
