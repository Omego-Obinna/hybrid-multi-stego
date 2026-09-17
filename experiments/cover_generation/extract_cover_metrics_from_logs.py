#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

WITHIN_RE = re.compile(
    r"run(?P<run_id>\d+)_"
    r"(?:(?P<corpus_tag>alice|pride)_)?"
    r"seed(?P<seed>\d+)_"
    r"(?P<kind>mauve|roberta)_"
    r"(?P<stream>gamma1|gamma2|human)",
    re.IGNORECASE,
)

MAUVE_RE = re.compile(rf"MAUVE:\s*(?P<value>{FLOAT_RE})")
ACC_RE = re.compile(rf"eval_accuracy:\s*(?P<value>{FLOAT_RE})")
AUC_RE = re.compile(rf"eval_roc_auc:\s*(?P<value>{FLOAT_RE})")


def infer_corpus(run_id, corpus_tag):
    if corpus_tag == "alice":
        return "alice"
    if corpus_tag == "pride":
        return "pride_prejudice"

    run_id = int(run_id)
    if run_id in (1, 2, 3):
        return "sherlock"
    if run_id in (4, 5, 6):
        return "alice"
    if run_id in (7, 8, 9):
        return "pride_prejudice"

    return "unknown"


def normalise_stream(stream):
    if stream == "human":
        return "human_reference"
    return stream


def comparison_type(stream):
    if stream == "human_reference":
        return "human_vs_test"
    return "selected_vs_test"


def parse_log_file(path):
    name = path.name
    m = WITHIN_RE.search(name)
    if not m:
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")

    run_id = int(m.group("run_id"))
    seed = int(m.group("seed"))
    corpus = infer_corpus(run_id, m.group("corpus_tag"))
    kind = m.group("kind").lower()
    stream = normalise_stream(m.group("stream").lower())

    key = (run_id, corpus, seed, stream, comparison_type(stream))

    out = {
        "run_id": run_id,
        "corpus": corpus,
        "seed": seed,
        "stream": stream,
        "comparison_type": comparison_type(stream),
        "mauve": None,
        "roberta_accuracy": None,
        "roberta_auc": None,
        "notes": "parsed from logs",
        "source_logs": str(path),
    }

    if kind == "mauve":
        mm = MAUVE_RE.search(text)
        if mm:
            out["mauve"] = float(mm.group("value"))

    elif kind == "roberta":
        am = ACC_RE.search(text)
        um = AUC_RE.search(text)
        if am:
            out["roberta_accuracy"] = float(am.group("value"))
        if um:
            out["roberta_auc"] = float(um.group("value"))

    return key, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--out_csv", default="results/cover_eval_summary_from_logs.csv")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    records = {}

    for path in sorted(log_dir.glob("*.log")):
        parsed = parse_log_file(path)
        if parsed is None:
            continue

        key, row = parsed

        if key not in records:
            records[key] = row
        else:
            existing = records[key]
            for field in ["mauve", "roberta_accuracy", "roberta_auc"]:
                if row.get(field) is not None:
                    existing[field] = row[field]
            existing["source_logs"] += f";{path}"

    rows = list(records.values())

    stream_order = {"gamma1": 1, "gamma2": 2, "human_reference": 3}
    rows = sorted(rows, key=lambda r: (r["run_id"], stream_order.get(r["stream"], 99)))

    df = pd.DataFrame(rows)

    expected_cols = [
        "run_id", "corpus", "seed", "stream", "comparison_type",
        "mauve", "roberta_accuracy", "roberta_auc",
        "notes", "source_logs",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df = df[expected_cols]
    df.to_csv(out_csv, index=False)

    print(f"[DONE] Parsed {len(df)} rows from {log_dir}")
    print(f"[DONE] Wrote: {out_csv}")

    missing = df[
        (df["comparison_type"] == "selected_vs_test")
        & (df[["mauve", "roberta_accuracy", "roberta_auc"]].isna().any(axis=1))
    ]

    if len(missing):
        missing_path = out_csv.parent / "cover_eval_summary_from_logs_missing.csv"
        missing.to_csv(missing_path, index=False)
        print(f"[WARN] Missing values found. See: {missing_path}")
    else:
        print("[OK] No missing selected-stream MAUVE/RoBERTa values detected.")


if __name__ == "__main__":
    main()
