#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, math, json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare

# Optional deps
try:
    import textstat
    HAS_TEXTSTAT = True
except Exception:
    HAS_TEXTSTAT = False

# --- NLTK: make tokenizers/tagger robust across versions ---
import nltk

def _soft_nltk_find_or_dl(resource_key: str, lookup_path: str):
    try:
        nltk.data.find(lookup_path)
    except LookupError:
        try:
            nltk.download(resource_key, quiet=True)
        except Exception:
            pass

# punkt (classic) and punkt_tab (newer NLTK) for sent/word tokenizers
_soft_nltk_find_or_dl("punkt", "tokenizers/punkt")
_soft_nltk_find_or_dl("punkt_tab", "tokenizers/punkt_tab/english")

# Taggers (name varies across versions)
_soft_nltk_find_or_dl("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger")
_soft_nltk_find_or_dl("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")

from nltk import word_tokenize, pos_tag

# --- sklearn (optional discriminability) ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def read_lines(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


def tokenize_words(s: str):
    # Keep only tokens with alnum somewhere (filters punctuation)
    return [w for w in word_tokenize(s) if any(ch.isalnum() for ch in w)]


def ttr(tokens):
    return (len(set(tokens)) / len(tokens)) if tokens else 0.0


def herdan_c(tokens):
    N = len(tokens)
    V = len(set(tokens))
    if N <= 1 or V <= 1:
        return 0.0
    return math.log(V) / math.log(N)


def ngram_counts(tokens, n=1, level="word"):
    """
    level="word": tokens is a list of word tokens.
    level="char": tokens is a list of word tokens; we convert to a single string with spaces
                  and then take character n-grams over that string.
    """
    if level == "word":
        seq = tokens
        grams = []
        for i in range(len(seq) - n + 1):
            grams.append(tuple(seq[i:i+n]))
        return Counter(grams)
    else:
        # char level
        joined = " ".join(tokens)
        grams = []
        L = len(joined)
        for i in range(L - n + 1):
            grams.append(joined[i:i+n])
        return Counter(grams)


def align_js(counter_p: Counter, counter_q: Counter) -> float:
    """Compute Jensen–Shannon divergence^2 (in bits) between two histograms."""
    keys = set(counter_p) | set(counter_q)
    if not keys:
        return 0.0
    p = np.array([counter_p.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([counter_q.get(k, 0.0) for k in keys], dtype=float)
    ps, qs = p.sum(), q.sum()
    if ps == 0 and qs == 0:
        return 0.0
    if ps > 0:
        p = p / ps
    if qs > 0:
        q = q / qs
    # jensenshannon returns a distance; square it to get divergence (base=2 -> bits)
    return float(jensenshannon(p, q, base=2) ** 2)


def pos_hist(sentences):
    tags = Counter()
    for s in sentences:
        toks = tokenize_words(s)
        if not toks:
            continue
        try:
            t = pos_tag(toks)
        except Exception:
            # Fallback (older APIs); if it still fails, skip tagging for this sentence
            try:
                t = nltk.pos_tag(toks)
            except Exception:
                t = []
        for _, tag in t:
            tags[tag] += 1
    return tags


def readability_scores(sentences):
    if not HAS_TEXTSTAT:
        return {}
    text = "\n".join(sentences)
    out = {}
    def _try(name, fn):
        try:
            out[name] = float(fn(text))
        except Exception:
            pass
    _try("flesch_reading_ease", textstat.flesch_reading_ease)
    _try("flesch_kincaid_grade", textstat.flesch_kincaid_grade)
    _try("smog_index", textstat.smog_index)
    _try("coleman_liau_index", textstat.coleman_liau_index)
    _try("automated_readability_index", textstat.automated_readability_index)
    _try("dale_chall_readability_score", textstat.dale_chall_readability_score)
    return out


def classifier_accuracy(sents1, sents2, max_samples=1000):
    if not HAS_SKLEARN:
        return None
    s1 = sents1[:max_samples]
    s2 = sents2[:max_samples]
    X_text = s1 + s2
    y = np.array([0]*len(s1) + [1]*len(s2))
    if len(X_text) < 10 or len(set(X_text)) < 4:
        return None
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2)
    try:
        X = vec.fit_transform(X_text)
    except Exception:
        return None
    if X.shape[0] < 10 or X.shape[1] < 2:
        return None
    clf = LogisticRegression(max_iter=1000)
    try:
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        return float(scores.mean()), float(scores.std())
    except Exception:
        return None


def make_hist(ax, data1, data2, label1, label2, bins=20, title="", xlabel=""):
    if not data1 and not data2:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    if data1:
        ax.hist(data1, bins=bins, alpha=0.5, label=label1, density=True)
    if data2:
        ax.hist(data2, bins=bins, alpha=0.5, label=label2, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()


def ecdf(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([0.0]), np.array([0.0])
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def safe_chisquare(obs_counts: np.ndarray, exp_counts: np.ndarray):
    """
    SciPy's chisquare requires equal totals. We scale expected to observed total,
    add a tiny smoothing to avoid zeros, then run the test.
    Returns (chi2_stat, p_value).
    """
    obs = np.asarray(obs_counts, dtype=float)
    exp = np.asarray(exp_counts, dtype=float)

    # Smoothing
    obs += 1.0
    exp += 1.0

    o_sum, e_sum = obs.sum(), exp.sum()
    if o_sum <= 0 or e_sum <= 0:
        return 0.0, 1.0
    exp_scaled = exp * (o_sum / e_sum)
    try:
        chi2, pval = chisquare(obs, f_exp=exp_scaled)
    except Exception:
        # Fallback: if something still odd, return neutral
        chi2, pval = 0.0, 1.0
    return float(chi2), float(pval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma1", default="gamma1.txt")
    ap.add_argument("--gamma2", default="gamma2.txt")
    ap.add_argument("--outdir", default="analysis_out")
    ap.add_argument("--title1", default="γ1")
    ap.add_argument("--title2", default="γ2")
    ap.add_argument("--benign", help="Path to benign.txt for external control")
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    g1 = read_lines(Path(args.gamma1))
    g2 = read_lines(Path(args.gamma2))

    # Tokenization
    tok1 = [tokenize_words(s) for s in g1]
    tok2 = [tokenize_words(s) for s in g2]
    flat1 = [w for sent in tok1 for w in sent]
    flat2 = [w for sent in tok2 for w in sent]

    # Basic stats
    sentlen1 = [len(s) for s in tok1]
    sentlen2 = [len(s) for s in tok2]
    wordlen1 = [len(w) for w in flat1]
    wordlen2 = [len(w) for w in flat2]

    # Richness
    ttr1, ttr2 = ttr(flat1), ttr(flat2)
    hc1, hc2 = herdan_c(flat1), herdan_c(flat2)

    # n-gram JS divergences
    results = {}
    for level in ("word", "char"):
        for n in (1, 2, 3):
            c1 = ngram_counts(flat1, n=n, level=level) if level == "word" else ngram_counts(flat1, n=n, level="char")
            c2 = ngram_counts(flat2, n=n, level=level) if level == "word" else ngram_counts(flat2, n=n, level="char")
            results[f"js_{level}_{n}"] = align_js(c1, c2)

    # POS tag distribution + χ²
    pos1 = pos_hist(g1)
    pos2 = pos_hist(g2)
    tags = sorted(set(pos1) | set(pos2))
    obs = np.array([pos1.get(t, 0) for t in tags], dtype=float)
    exp = np.array([pos2.get(t, 0) for t in tags], dtype=float)
    pos_chi2, pos_p = safe_chisquare(obs, exp)

    # Readability (optional)
    read1 = readability_scores(g1)
    read2 = readability_scores(g2)

    # Discriminability (optional)
    clf = classifier_accuracy(g1, g2)
    clf_mean, clf_std = (clf if clf is not None else (None, None))

    # Save metrics
    metrics = {
        "n_sent_1": len(g1), "n_sent_2": len(g2),
        "n_tokens_1": len(flat1), "n_tokens_2": len(flat2),
        "sent_len_mean_1": float(np.mean(sentlen1)) if sentlen1 else 0.0,
        "sent_len_mean_2": float(np.mean(sentlen2)) if sentlen2 else 0.0,
        "word_len_mean_1": float(np.mean(wordlen1)) if wordlen1 else 0.0,
        "word_len_mean_2": float(np.mean(wordlen2)) if wordlen2 else 0.0,
        "ttr_1": ttr1, "ttr_2": ttr2,
        "herdan_c_1": hc1, "herdan_c_2": hc2,
        "pos_chi2_stat": pos_chi2,
        "pos_chi2_p": pos_p,
        "clf_char_ngram_acc_mean": clf_mean,
        "clf_char_ngram_acc_std": clf_std
    }
    metrics.update({f"read_{k}_1": v for k, v in read1.items()})
    metrics.update({f"read_{k}_2": v for k, v in read2.items()})
    metrics.update(results)

    df = pd.DataFrame([metrics])
    df.to_csv(outdir / "indistinguishability_metrics.csv", index=False)
    with open(outdir / "indistinguishability_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Plots ---
    plt.rcParams.update({"figure.figsize": (7, 5), "figure.dpi": 120, "font.size": 11})

    # 1) Sentence length
    fig1, ax1 = plt.subplots()
    make_hist(ax1, sentlen1, sentlen2, args.title1, args.title2,
              bins=20, title="Sentence length distribution", xlabel="Words per sentence")
    fig1.tight_layout()
    fig1.savefig(outdir / "sent_length_hist.png")
    plt.close(fig1)

    # 2) Word length
    fig2, ax2 = plt.subplots()
    make_hist(ax2, wordlen1, wordlen2, args.title1, args.title2,
              bins=20, title="Word length distribution", xlabel="Characters per word")
    fig2.tight_layout()
    fig2.savefig(outdir / "word_length_hist.png")
    plt.close(fig2)

    # 3) CDFs (sentence length)
    x1, y1 = ecdf(sentlen1)
    x2, y2 = ecdf(sentlen2)
    fig3, ax3 = plt.subplots()
    ax3.plot(x1, y1, label=f"{args.title1} CDF")
    ax3.plot(x2, y2, label=f"{args.title2} CDF")
    ax3.set_title("Sentence length CDF")
    ax3.set_xlabel("Words per sentence")
    ax3.set_ylabel("Cumulative probability")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(outdir / "sent_length_cdf.png")
    plt.close(fig3)

    # 4) POS bar chart (top 20)
    pos_total = pos1 + pos2
    top_tags = [t for t, _ in pos_total.most_common(20)]
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(top_tags))
    ax4.bar(x - 0.2, [pos1.get(t, 0) for t in top_tags], width=0.4, label=args.title1)
    ax4.bar(x + 0.2, [pos2.get(t, 0) for t in top_tags], width=0.4, label=args.title2)
    ax4.set_xticks(x); ax4.set_xticklabels(top_tags, rotation=45, ha="right")
    ax4.set_title(f"POS tag counts (top 20), χ²≈{pos_chi2:.2f}, p≈{pos_p:.3g}")
    ax4.set_ylabel("Count")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(outdir / "pos_bars.png")
    plt.close(fig4)

    # 5) JS divergences summary
    labels = []
    values = []
    for k in sorted([k for k in results if k.startswith("js_")]):
        labels.append(k.replace("js_", "").replace("_", "-"))
        values.append(results[k])
    fig5, ax5 = plt.subplots()
    ax5.bar(range(len(labels)), values)
    ax5.set_xticks(range(len(labels))); ax5.set_xticklabels(labels, rotation=45, ha="right")
    ax5.set_ylabel("JS divergence (bits)")
    ax5.set_title("n-gram JS divergence (lower is closer)")
    fig5.tight_layout()
    fig5.savefig(outdir / "js_divergences.png")
    plt.close(fig5)

    # 6) Optional readability comparison
    if HAS_TEXTSTAT and read1 and read2:
        keys = sorted(set(read1.keys()) & set(read2.keys()))
        if keys:
            fig6, ax6 = plt.subplots()
            idx = np.arange(len(keys))
            ax6.bar(idx - 0.2, [read1[k] for k in keys], width=0.4, label=args.title1)
            ax6.bar(idx + 0.2, [read2[k] for k in keys], width=0.4, label=args.title2)
            ax6.set_xticks(idx); ax6.set_xticklabels(keys, rotation=45, ha="right")
            ax6.set_title("Readability metrics (note: scale varies per metric)")
            ax6.legend()
            fig6.tight_layout()
            fig6.savefig(outdir / "readability.png")
            plt.close(fig6)

    # Consolidated PDF (best-effort)
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = outdir / "indistinguishability_report.pdf"
        with PdfPages(pdf_path) as pdf:
            for img in [
                "sent_length_hist.png",
                "word_length_hist.png",
                "sent_length_cdf.png",
                "pos_bars.png",
                "js_divergences.png",
                "readability.png",
            ]:
                p = outdir / img
                if p.exists():
                    fig = plt.figure(figsize=(8, 6))
                    im = plt.imread(p)
                    plt.imshow(im); plt.axis("off")
                    pdf.savefig(fig); plt.close(fig)
        print(f"[OK] Wrote consolidated PDF: {pdf_path}")
    except Exception as e:
        print(f"[WARN] Could not create consolidated PDF: {e}")

    print("[OK] Saved metrics JSON/CSV and plots in:", outdir)


if __name__ == "__main__":
    main()

