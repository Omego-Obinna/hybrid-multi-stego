#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import evaluate
import torch

def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]

def make_ds(positives, negatives):  # binary: pos=gamma, neg=benign
    texts = negatives + positives
    labels = [0]*len(negatives) + [1]*len(positives)
    return Dataset.from_dict({"text": texts, "label": labels})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", required=True, help="Path to gamma*.txt (γ1 or γ2)")
    ap.add_argument("--benign", required=True, help="Path to benign.txt")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    pos = load_txt(Path(args.gamma))
    neg = load_txt(Path(args.benign))

    # simple split (stratified)
    pos_train, pos_val = train_test_split(pos, test_size=0.2, random_state=42)
    neg_train, neg_val = train_test_split(neg, test_size=0.2, random_state=42)

    ds = DatasetDict({
        "train": make_ds(pos_train, neg_train),
        "validation": make_ds(pos_val, neg_val)
    })

    tok = AutoTokenizer.from_pretrained(args.model)
    def tok_fn(batch): return tok(batch["text"], truncation=True, padding="max_length", max_length=256)
    ds = ds.map(tok_fn, batched=True)
    ds = ds.remove_columns(["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    metric_acc = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:,1]
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, np.argmax(logits, axis=-1))
        m = metric_acc.compute(predictions=np.argmax(logits, axis=-1), references=labels)
        m.update({"roc_auc": float(auc), "accuracy": float(acc)})
        return m

    args_tr = TrainingArguments(
        output_dir="roberta_det_out",
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\n== Eval metrics ==")
    for k,v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()

