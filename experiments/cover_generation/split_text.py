# split_text.py
import argparse, random
from pathlib import Path

def read_lines(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--test_out", required=True)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lines = read_lines(args.infile)
    random.Random(args.seed).shuffle(lines)

    n_test = int(len(lines) * args.test_frac)
    test = lines[:n_test]
    train = lines[n_test:]

    Path(args.train_out).write_text("\n".join(train) + "\n", encoding="utf-8")
    Path(args.test_out).write_text("\n".join(test) + "\n", encoding="utf-8")

    print(f"Train/reference lines: {len(train)}")
    print(f"Test lines: {len(test)}")

if __name__ == "__main__":
    main()
