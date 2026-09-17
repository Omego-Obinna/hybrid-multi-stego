import argparse
import mauve


def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", required=True, help="path to first text file, e.g., gamma1.txt")
    ap.add_argument("--q", required=True, help="path to second text file, e.g., benign.txt")
    ap.add_argument("--device", type=int, default=0, help="GPU id; use -1 for CPU")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--num_buckets", type=int, default=50)
    args = ap.parse_args()

    p_text = read_lines(args.p)
    q_text = read_lines(args.q)

    print(f"Loaded p: {len(p_text)} lines")
    print(f"Loaded q: {len(q_text)} lines")
    print(f"Using num_buckets: {args.num_buckets}")

    out = mauve.compute_mauve(
        p_text=p_text,
        q_text=q_text,
        device_id=args.device,
        max_text_length=args.max_len,
        num_buckets=args.num_buckets,
        verbose=True
    )

    print("MAUVE:", float(out.mauve))
