#!/usr/bin/env python3
import argparse, hashlib, hmac, os, random, sys, glob
from pathlib import Path
import markovify, nltk, unidecode

# Ensure NLTK punkt is available (silent if already present)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

def derive_seed(key: str, label: bytes) -> int:
    # Deterministic 32-bit seed from key+label (HMAC-SHA256)
    d = hmac.new(key.encode("utf-8"), label, hashlib.sha256).digest()
    return int.from_bytes(d[:4], "big")

def read_corpus(corpus_dir: Path) -> str:
    files = []
    for ext in ("*.txt",):
        files.extend(glob.glob(str(corpus_dir / ext)))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    texts = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            texts.append(unidecode.unidecode(fh.read()))
    return "\n\n".join(texts)

def build_model(text: str, state_size: int = 2):
    # Markov model over sentences
    return markovify.Text(text, state_size=state_size, well_formed=True)

def generate_stream(model, n_sent, min_len, max_len, rng_seed, tries=200):
    # Set Python RNG so generation is deterministic per stream
    random.seed(rng_seed)
    out = []
    for _ in range(n_sent):
        sent = None
        for _ in range(tries):
            s = model.make_sentence(tries=100)
            if not s:
                continue
            wc = len(s.split())
            if wc < min_len or wc > max_len:
                continue
            sent = s
            break
        if not sent:
            # fallback short sentence if model fails
            sent = model.make_short_sentence(max_len*8) or "..."
        out.append(sent)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Directory with .txt corpus files")
    ap.add_argument("--key", required=True, help="Shared session key (string)")
    ap.add_argument("--n_sent", type=int, default=60)
    ap.add_argument("--min_len", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=18)
    ap.add_argument("--out1", required=True)
    ap.add_argument("--out2", required=True)
    ap.add_argument("--state_size", type=int, default=2)
    args = ap.parse_args()

    text = read_corpus(Path(args.corpus))
    model = build_model(text, state_size=args.state_size)

    seed1 = derive_seed(args.key, b"gamma1")
    seed2 = derive_seed(args.key, b"gamma2")

    g1 = generate_stream(model, args.n_sent, args.min_len, args.max_len, seed1)
    g2 = generate_stream(model, args.n_sent, args.min_len, args.max_len, seed2)

    with open(args.out1, "w", encoding="utf-8") as f1:
        f1.write("\n".join(g1) + "\n")
    with open(args.out2, "w", encoding="utf-8") as f2:
        f2.write("\n".join(g2) + "\n")

if __name__ == "__main__":
    main()
