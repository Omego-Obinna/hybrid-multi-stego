#!/usr/bin/env python3
"""Analyse Ours-QIM mask-only ablations and generate publication outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "ones_ratio",
    "absolute_bias",
    "min_entropy_per_bit",
    "serial_correlation_lag1",
    "transition_rate",
    "longest_run",
    "approximate_entropy_m2",
    "mean_block_bias",
]


def classifier_results(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config_id, payload, message_class), group in features.groupby(
        ["config_id", "payload_bpp", "message_class"]
    ):
        folds = []
        for test_seed in sorted(group["seed"].unique()):
            train = group[group["seed"] != test_seed]
            test = group[group["seed"] == test_seed]
            if train["label"].nunique() < 2 or test["label"].nunique() < 2:
                continue

            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=20260605,
                        ),
                    ),
                ]
            )
            model.fit(train[FEATURES], train["label"])
            pred = model.predict(test[FEATURES])
            score = model.predict_proba(test[FEATURES])[:, 1]
            folds.append(
                (
                    accuracy_score(test["label"], pred),
                    roc_auc_score(test["label"], score),
                )
            )

        if folds:
            accuracy = float(np.mean([item[0] for item in folds]))
            auc = float(np.mean([item[1] for item in folds]))
            rows.append(
                {
                    "config_id": config_id,
                    "payload_bpp": float(payload),
                    "message_class": message_class,
                    "accuracy": accuracy,
                    "auc": auc,
                    "advantage_half": abs(accuracy - 0.5),
                    "epsilon_param_proxy_norm": 2.0 * abs(accuracy - 0.5),
                    "folds": len(folds),
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    masked = metrics[metrics["stream"] == "b"].copy()
    group_cols = [
        "config_id",
        "psi_mode",
        "dither_mode",
        "delta_scale",
        "delta",
        "payload_bpp",
        "message_class",
    ]
    return (
        masked.groupby(group_cols)
        .agg(
            samples=("n_bits", "size"),
            bits=("n_bits", "sum"),
            ones_ratio_mean=("ones_ratio", "mean"),
            absolute_bias_mean=("absolute_bias", "mean"),
            min_entropy_mean=("min_entropy_per_bit", "mean"),
            serial_corr_mean=("serial_correlation_lag1", "mean"),
            transition_rate_mean=("transition_rate", "mean"),
            approx_entropy_mean=("approximate_entropy_m2", "mean"),
            monobit_pass_rate=("monobit_p", lambda s: float((s >= 0.01).mean())),
            runs_pass_rate=("runs_p", lambda s: float((s >= 0.01).mean())),
            block_frequency_pass_rate=(
                "block_frequency_p",
                lambda s: float((s >= 0.01).mean()),
            ),
        )
        .reset_index()
    )


def write_summary_table(data: pd.DataFrame, path: Path) -> None:
    zero = data[data["message_class"] == "zero"].sort_values(
        ["payload_bpp", "psi_mode", "dither_mode", "delta_scale"]
    )
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Ours-QIM masking ablations for the all-zero message class, for which $b=\mu$. Values are aggregated across covers and seeds. Pass rates use a descriptive threshold of $0.01$ and are not cryptographic certification.}",
        r"\label{tab:qim_mask_ablation}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"$\psi(\cdot)$ & Dither & $\Delta/\Delta_0$ & Payload & Bias & $H_{\infty}$ & Monobit pass & Runs pass \\",
        r"\midrule",
    ]
    for _, row in zero.iterrows():
        line = (
            f"{row['psi_mode']} & {row['dither_mode']} & "
            f"{row['delta_scale']:.1f} & {row['payload_bpp']:.2f} & "
            f"{row['absolute_bias_mean']:.4f} & {row['min_entropy_mean']:.4f} & "
            f"{100.0 * row['monobit_pass_rate']:.1f}\\% & "
            f"{100.0 * row['runs_pass_rate']:.1f}\\% " + r"\\"
        )
        lines.append(line)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_classifier_table(data: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Detector-dependent distinguishability of masked-payload windows from independent uniform windows under leave-one-seed-out evaluation. $\widehat{\varepsilon}_{\mathsf{param}}^{\mathsf{clf}}=2|\widehat a-\tfrac12|$ is a normalised empirical proxy, not a certified upper bound on total variation.}",
        r"\label{tab:qim_mask_classifier}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Configuration & Payload & Message class & Accuracy & AUC & Half-scale adv. & $\widehat{\varepsilon}_{\mathsf{param}}^{\mathsf{clf}}$ \\",
        r"\midrule",
    ]
    for _, row in data.sort_values(
        ["payload_bpp", "message_class", "config_id"]
    ).iterrows():
        line = (
            f"{row['config_id']} & {row['payload_bpp']:.2f} & "
            f"{row['message_class']} & {row['accuracy']:.4f} & "
            f"{row['auc']:.4f} & {row['advantage_half']:.4f} & "
            f"{row['epsilon_param_proxy_norm']:.4f} " + r"\\"
        )
        lines.append(line)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figures(
    summary: pd.DataFrame,
    classifier: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zero = summary[
        (summary["message_class"] == "zero")
        & (summary["dither_mode"] == "cover_prf")
    ]

    for payload in sorted(zero["payload_bpp"].unique()):
        data = zero[zero["payload_bpp"] == payload]
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        for psi_mode, group in data.groupby("psi_mode"):
            group = group.sort_values("delta_scale")
            ax.plot(
                group["delta_scale"],
                group["min_entropy_mean"],
                marker="o",
                label=psi_mode,
            )
        ax.set_xlabel(r"$\Delta/\Delta_0$")
        ax.set_ylabel(r"Empirical $H_{\infty}$ per bit")
        ax.set_title(f"Ours-QIM mask min-entropy at {payload:.2f} bpp")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        for extension in formats:
            fig.savefig(
                out_dir
                / f"fig_mask_min_entropy_p{int(round(payload * 100)):03d}.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(fig)

    if not classifier.empty:
        zero_clf = classifier[classifier["message_class"] == "zero"].copy()
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        x = np.arange(len(zero_clf))
        ax.bar(x, zero_clf["epsilon_param_proxy_norm"])
        ax.set_xticks(x, zero_clf["config_id"], rotation=90)
        ax.set_ylabel(r"$\widehat{\varepsilon}_{\mathsf{param}}^{\mathsf{clf}}$")
        ax.set_title("Classifier-based masked-payload distinguishability")
        ax.grid(axis="y", alpha=0.3)
        for extension in formats:
            fig.savefig(
                out_dir / f"fig_mask_classifier_proxy.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(fig)


def write_draft(path: Path) -> None:
    path.write_text(
        r"""\subsection{Masking Ablations and Payload Pseudorandomness}
\label{supp-subsec:masking_ablations}

This experiment evaluates the formal Ours-QIM masking layer without invoking image embedding. For each cover $o$, the public feature $\eta=\psi(o)$ conditions a keyed dither, and the resulting QIM mask $\mu$ is combined with the message as $b=m\oplus\mu$. The ablation varies $\psi(\cdot)$, the dither source, and the relative QIM step $\Delta/\Delta_0$.

\input{tables/table_qim_mask_ablation}
\input{tables/table_qim_mask_classifier}

For the all-zero message, $b=\mu$, so the measurements expose the mask distribution directly. Structured text, alternating, low-entropy, and random messages assess whether the mask suppresses message-dependent structure. The classifier proxy is $\widehat{\varepsilon}_{\mathsf{param}}^{\mathsf{clf}}=2|\widehat a-\tfrac12|$. Values near zero indicate chance-level distinguishability from independent uniform windows. This detector-dependent quantity is not a certified upper bound on the total-variation term $\varepsilon_{\mathsf{param}}$ in the composition theorem. Statistical tests and min-entropy estimates are descriptive diagnostics and do not prove cryptographic pseudorandomness.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png", "svg"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_metrics(pd.read_csv(args.metrics_csv))
    classifier = classifier_results(pd.read_csv(args.features_csv))

    summary.to_csv(tables_dir / "qim_mask_ablation_summary.csv", index=False)
    classifier.to_csv(
        tables_dir / "qim_mask_classifier_results.csv",
        index=False,
    )
    write_summary_table(summary, tables_dir / "table_qim_mask_ablation.tex")
    write_classifier_table(
        classifier,
        tables_dir / "table_qim_mask_classifier.tex",
    )
    save_figures(summary, classifier, figures_dir, args.formats, args.dpi)
    write_draft(out_dir / "masking_ablations_draft.tex")

    metadata = {
        "classifier": "StandardScaler + L2 logistic regression",
        "validation": "leave-one-seed-out",
        "normalised_proxy": "2*abs(accuracy-0.5)",
        "half_scale_advantage": "abs(accuracy-0.5)",
    }
    (out_dir / "mask_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
