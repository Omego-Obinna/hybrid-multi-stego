#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

LOG_RE = re.compile(
    r"^ext(?P<external_run_id>\d+)_"
    r"(?P<label>.+?)_"
    r"(?P<kind>mauve|roberta)_"
    r"(?P<stream>gamma1|gamma2|human)_vs_test\.log$",
    re.IGNORECASE,
)

MAUVE_RE = re.compile(rf"MAUVE:\s*(?P<value>{FLOAT_RE})")
ACC_RE = re.compile(rf"eval_accuracy:\s*(?P<value>{FLOAT_RE})")
AUC_RE = re.compile(rf"eval_roc_auc:\s*(?P<value>{FLOAT_RE})")
SEED_RE = re.compile(r"selected_seed(?P<seed>\d+)")


LABEL_MAP = {
    "alice_pride_to_sherlock": {
        "reference_books": "alice+pride_prejudice",
        "test_book": "sherlock",
        "description": "Alice + Pride and Prejudice → Sherlock",
    },
    "alice_sherlock_to_pride": {
        "reference_books": "alice+sherlock",
        "test_book": "pride_prejudice",
        "description": "Alice + Sherlock → Pride and Prejudice",
    },
    "pride_sherlock_to_alice": {
        "reference_books": "pride_prejudice+sherlock",
        "test_book": "alice",
        "description": "Pride and Prejudice + Sherlock → Alice",
    },
}


def normalise_stream(stream):
    if stream.lower() == "human":
        return "human_reference"
    return stream.lower()


def comparison_type(stream):
    return "human_vs_test" if stream == "human_reference" else "selected_vs_test"


def parse_one_log(path: Path):
    m = LOG_RE.match(path.name)
    if not m:
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")

    external_run_id = int(m.group("external_run_id"))
    label = m.group("label")
    kind = m.group("kind").lower()
    stream = normalise_stream(m.group("stream"))

    info = LABEL_MAP.get(label, {
        "reference_books": "unknown",
        "test_book": "unknown",
        "description": label,
    })

    seed_match = SEED_RE.search(text)
    seed = int(seed_match.group("seed")) if seed_match else 42

    row = {
        "external_run_id": external_run_id,
        "label": label,
        "description": info["description"],
        "reference_books": info["reference_books"],
        "test_book": info["test_book"],
        "seed": seed,
        "stream": stream,
        "comparison_type": comparison_type(stream),
        "mauve": None,
        "roberta_accuracy": None,
        "roberta_auc": None,
        "notes": "leave-one-book-out; parsed from logs",
        "source_logs": str(path),
    }

    if kind == "mauve":
        mm = MAUVE_RE.search(text)
        if mm:
            row["mauve"] = float(mm.group("value"))

    elif kind == "roberta":
        am = ACC_RE.search(text)
        um = AUC_RE.search(text)

        if am:
            row["roberta_accuracy"] = float(am.group("value"))
        if um:
            row["roberta_auc"] = float(um.group("value"))

    key = (
        external_run_id,
        label,
        stream,
        row["comparison_type"],
    )

    return key, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--out_csv", default="results/external_cover_eval_summary_from_logs.csv")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    records = {}

    for path in sorted(log_dir.glob("ext*.log")):
        parsed = parse_one_log(path)
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

    df = pd.DataFrame(records.values())

    if df.empty:
        print("[WARN] No external validation logs were parsed.")
        return

    stream_order = {
        "gamma1": 1,
        "gamma2": 2,
        "human_reference": 3,
    }

    df["stream_order"] = df["stream"].map(stream_order).fillna(99)
    df = df.sort_values(["external_run_id", "stream_order"]).drop(columns=["stream_order"])

    expected_cols = [
        "external_run_id",
        "label",
        "description",
        "reference_books",
        "test_book",
        "seed",
        "stream",
        "comparison_type",
        "mauve",
        "roberta_accuracy",
        "roberta_auc",
        "notes",
        "source_logs",
    ]

    df = df[expected_cols]
    df.to_csv(out_csv, index=False)

    print(f"[DONE] Parsed {len(df)} external-validation rows.")
    print(f"[DONE] Wrote: {out_csv}")

    missing = df[
        (
            (df["comparison_type"] == "selected_vs_test")
            & df[["mauve", "roberta_accuracy", "roberta_auc"]].isna().any(axis=1)
        )
        |
        (
            (df["comparison_type"] == "human_vs_test")
            & df[["mauve"]].isna().any(axis=1)
        )
    ]

    missing_path = out_csv.parent / "external_cover_eval_missing_from_logs.csv"
    missing.to_csv(missing_path, index=False)

    if len(missing):
        print(f"[WARN] Some values are missing. See: {missing_path}")
    else:
        print("[OK] No missing external-validation metrics detected.")

    # Summary table: selected streams only
    selected = df[df["comparison_type"] == "selected_vs_test"].copy()

    summary = selected.groupby(["test_book", "stream"])[
        ["mauve", "roberta_accuracy", "roberta_auc"]
    ].agg(["mean", "std", "count"]).reset_index()

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    summary_path = out_csv.parent / "external_cover_eval_summary_table.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[DONE] Wrote summary table: {summary_path}")

    try:
        tex_path = out_csv.parent / "external_cover_eval_summary_table.tex"
        summary.to_latex(tex_path, index=False, float_format="%.4f")
        print(f"[DONE] Wrote LaTeX table: {tex_path}")
    except Exception as e:
        print(f"[WARN] Could not write LaTeX table: {e}")


if __name__ == "__main__":
    main()
