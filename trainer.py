#!/usr/bin/env python
"""
trainer.py
==========

Fetch GitHub issues, curate clean training data, and train one of three
models:

  • baseline   – LogisticRegression on TF-IDF (quick sanity check)
  • embed      – MiniLM sentence-embeddings  →  LogisticRegression   ← recommended
  • finetune   – DistilBERT fine-tuned on your bug dataset

Run:
    python trainer.py --model embed        # best speed/accuracy trade-off
    python trainer.py --model finetune     # highest potential accuracy

Dependencies (add to requirements.txt):
    transformers sentence-transformers torch scikit-learn requests pandas python-dotenv
"""
from __future__ import annotations
import argparse, os, time, pickle, json
from pathlib import Path
from typing import List
import requests, pandas as pd
from dotenv import load_dotenv

# ───────────────────────── config ──────────────────────────
REPOS               = ["microsoft/vscode", "tensorflow/tensorflow"]
ISSUES_PER_REPO     = 500
OUT_DIR             = Path("model")
DATA_PATH           = Path("data/github_issues_clean.csv")
MAX_ISSUE_LEN       = 30_000            # ignore mega-issues
MIN_TEXT_LEN        = 20                # filter near-empty
SEED                = 42
# ───────────────────────────────────────────────────────────

load_dotenv()
HEADERS = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}" } if os.getenv("GITHUB_TOKEN") else {}

# ───────────────────────── helpers ─────────────────────────
LABEL_MAP = {
    2: ["p0", "severity: critical", "blocker", "priority: high", "priority: 0"],
    1: ["p1", "severity: major",    "priority: 1",  "priority: medium"],
    0: ["p2", "p3", "severity: minor", "trivial", "priority: low"]
}
KW_MAP = {
    2: ["crash", "fatal", "data loss", "security", "urgent"],
    1: ["error", "exception", "memory", "leak", "freeze", "slow", "hang"],
    0: []
}
def prio_from_labels(labels: List[dict]) -> int|None:
    names = [l["name"].lower() for l in labels]
    for p, pats in LABEL_MAP.items():
        if any(pat in n for n in names for pat in pats):
            return p
    return None
def prio_from_text(text: str) -> int:
    t = text.lower()
    for p, words in KW_MAP.items():
        if any(w in t for w in words):
            return p
    return 0
def fetch(repo: str, n: int):
    issues, page = [], 1
    while len(issues) < n:
        url = f"https://api.github.com/repos/{repo}/issues"
        r = requests.get(url, headers=HEADERS,
                         params={"state": "open", "per_page": 100, "page": page},
                         timeout=30)
        if r.status_code == 422:
            print(f"⚠️  Skipping page {page} of {repo} — likely no more issues.")
            break

        r.raise_for_status()
        batch = [i for i in r.json() if "pull_request" not in i]
        
        if not batch or len(batch) == 0: break

        issues += batch; page += 1; time.sleep(0.6)
    return issues[:n]

# ───────────────────────── data prep ───────────────────────
def build_dataset() -> pd.DataFrame:
    rows = []
    for repo in REPOS:
        for iss in fetch(repo, ISSUES_PER_REPO):
            prio = prio_from_labels(iss["labels"]) or prio_from_text(iss["title"]+" "+(iss.get("body") or ""))
            rows.append({
                "text": (iss["title"] or "") + " " + (iss.get("body") or ""),
                "priority": prio
            })
    df = pd.DataFrame(rows)
    df = df[df.text.str.len().between(MIN_TEXT_LEN, MAX_ISSUE_LEN)]
    DATA_PATH.parent.mkdir(exist_ok=True); df.to_csv(DATA_PATH, index=False)
    print("✅ dataset:", df.priority.value_counts().to_dict())
    return df

# ───────────────────────── models ──────────────────────────
def train_baseline(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1,3), stop_words="english", max_features=20_000)
    X = vec.fit_transform(df.text)
    y = df.priority
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X,y)
    return {"model": clf, "vectorizer": vec, "type": "tfidf"}

def train_embed(df):
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    X = emb.encode(df.text.tolist(), batch_size=64, show_progress_bar=True)
    y = df.priority
    clf = LogisticRegression(max_iter=4000, class_weight="balanced").fit(X,y)
    return {"model": clf, "embedder": "all-MiniLM-L6-v2", "type": "st"}

def train_finetune(df):
    # lightweight DistilBERT fine-tune (≈ 5-10 min on GPU Colab / ≈30 on CPU)
    from datasets import Dataset, load_metric
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds  = Dataset.from_pandas(df[["text","priority"]])
    ds  = ds.map(lambda e: tok(e["text"], truncation=True), batched=True)
    mod = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3)
    args = TrainingArguments("finetuned-bugbert", num_train_epochs=2,
                             per_device_train_batch_size=16,
                             learning_rate=2e-5, seed=SEED,
                             logging_steps=50, save_strategy="no")
    trainer = Trainer(model=mod, args=args, train_dataset=ds)
    trainer.train()
    return {"model": mod, "tokenizer": tok, "type": "bert"}

# ───────────────────────── save artefacts ──────────────────
def persist(bundle: dict):
    OUT_DIR.mkdir(exist_ok=True)
    meta = {"type": bundle["type"]}
    if bundle["type"] == "tfidf":
        pickle.dump(bundle["model"], open(OUT_DIR/"clf.pkl","wb"))
        pickle.dump(bundle["vectorizer"], open(OUT_DIR/"vec.pkl","wb"))
    elif bundle["type"] == "st":
        pickle.dump(bundle["model"], open(OUT_DIR/"clf.pkl","wb"))
        meta["embedder"] = bundle["embedder"]
    elif bundle["type"] == "bert":
        bundle["model"].save_pretrained(OUT_DIR/"bert_model")
        bundle["tokenizer"].save_pretrained(OUT_DIR/"bert_tok")
    json.dump(meta, open(OUT_DIR/"meta.json","w"))
    print("💾 saved artefacts to", OUT_DIR)

# ─────────────────────── main entry ────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline","embed","finetune"], default="embed")
    args = p.parse_args()

    df = build_dataset()
    if args.model=="baseline": bundle=train_baseline(df)
    elif args.model=="embed":  bundle=train_embed(df)
    else:                       bundle=train_finetune(df)
    persist(bundle)
