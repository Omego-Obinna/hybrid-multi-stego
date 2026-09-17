#!/usr/bin/env python3
"""Summarise existing Ours-QIM extraction correctness from prior logs."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-root', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    root = Path(args.log_root).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in sorted(root.rglob('embed_log_ours_qim.csv')):
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            use = [c for c in ['payload_bpp', 'success_extract', 'ber', 'ok'] if c in header]
            if 'payload_bpp' not in use or 'ber' not in use:
                continue
            df = pd.read_csv(path, usecols=use, low_memory=False)
            df['source'] = str(path)
            frames.append(df)
        except Exception as exc:
            print(f'[WARN] {path}: {exc}')

    if not frames:
        print('[WARN] No usable detailed logs found. Use the existing publication fidelity/runtime table.')
        return 0

    df = pd.concat(frames, ignore_index=True)
    df['payload_bpp'] = pd.to_numeric(df['payload_bpp'], errors='coerce')
    df['ber'] = pd.to_numeric(df['ber'], errors='coerce')
    if 'ok' in df:
        ok = df['ok'].astype(str).str.lower()
        df = df[ok.isin(['true', '1', 'yes']) | df['ok'].isna()]
    if 'success_extract' in df:
        success = df['success_extract'].astype(str).str.lower()
        df['success_extract_numeric'] = success.isin(['true', '1', 'yes']).astype(int)
    else:
        df['success_extract_numeric'] = (df['ber'] == 0).astype(int)

    summary = (
        df.groupby('payload_bpp')
        .agg(
            images=('ber', 'size'),
            extraction_success_rate=('success_extract_numeric', 'mean'),
            mean_ber=('ber', 'mean'),
            exact_recovery_rate=('ber', lambda s: float((s == 0).mean())),
        )
        .reset_index()
        .sort_values('payload_bpp')
    )
    summary.to_csv(out / 'qim_existing_recovery_summary.csv', index=False)

    lines = [
        r'\begin{table}[!t]',
        r'\centering',
        r'\caption{Existing Ours-QIM extraction correctness used to support recovery of $b\prime$ for the deployed configuration.}',
        r'\label{tab:qim_existing_recovery}',
        r'\begin{tabular}{rrrr}',
        r'\toprule',
        r'Payload & Images & Mean BER & Exact recovery \\',
        r'\midrule',
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['payload_bpp']:.2f} & {int(row['images'])} & "
            f"{row['mean_ber']:.4g} & {100.0 * row['exact_recovery_rate']:.2f}\\% "
            + r'\\'
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (out / 'table_qim_existing_recovery.tex').write_text('\n'.join(lines), encoding='utf-8')

    print(summary.to_string(index=False))
    print('[OK] No new embedding or extraction was performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
