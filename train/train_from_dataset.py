"""Train the CRF from the BIO dataset and compare against the shipped model.

Reads `train/data/<lang>.jsonl`, collapses BIO -> K/O (the shipped
`extract_keyword` joins contiguous `K` spans, so K/O stays drop-in), holds out a
test split, fits a fresh CRF, and reports exact-match + token-level keyword F1
for BOTH the shipped model and the freshly trained one on the same test set.

New models are written to `train/out/kx_<lang>.pkl` — NOT over the shipped
`crf_query_xtract/kx_*.pkl`. Promote only after reviewing the numbers.

Run::

    python train/train_from_dataset.py --langs en,pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

from sklearn_crfsuite import CRF

from crf_query_xtract import SearchtermExtractorCRF

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
SHIPPED = os.path.join(os.path.dirname(HERE), "crf_query_xtract")


def load(lang: str) -> List[dict]:
    path = os.path.join(DATA, f"{lang}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_gold(lang: str) -> List[dict]:
    path = os.path.join(DATA, "gold", f"{lang}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def keyword_of(labels: List[str], tokens: List[str]) -> str:
    """The shipped join: contiguous non-O tokens -> ' '.join (first span)."""
    spans, cur = [], []
    for tok, lab in zip(tokens, labels):
        if lab != "O":
            cur.append(tok)
        elif cur:
            spans.append(" ".join(cur))
            cur = []
    if cur:
        spans.append(" ".join(cur))
    return spans[0] if spans else ""


def evaluate(model: SearchtermExtractorCRF, rows: List[dict]) -> Dict[str, float]:
    exact = tp = fp = fn = 0
    for r in rows:
        gold = set(r["keyword"].lower().split())
        pred = model.extract_keyword(r["text"])
        pred_set = set(pred.lower().split())
        if pred.lower() == r["keyword"].lower():
            exact += 1
        tp += len(gold & pred_set)
        fp += len(pred_set - gold)
        fn += len(gold - pred_set)
    n = len(rows) or 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"exact": exact / n, "f1": f1, "n": len(rows)}


def fit(lang: str, train_rows: List[dict]) -> SearchtermExtractorCRF:
    base = SearchtermExtractorCRF(lang)  # for _sent2features + tagger
    X, y = [], []
    for r in train_rows:
        feats = base._sent2features(list(zip(r["tokens"], r["pos"])))
        labels = ["K" if l != "O" else "O" for l in r["labels"]]
        X.append(feats)
        y.append(labels)
    crf = CRF(algorithm="lbfgs", max_iterations=200, c1=0.1, c2=0.1, all_possible_transitions=True)
    crf.fit(X, y)
    base.model = crf
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="ca,da,de,en,es,eu,fr,gl,it,nl,pt")
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="held-out fraction used only when no gold split exists")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(args.seed)
    import joblib

    # The extractor runs AFTER intent classification, so it only ever sees search
    # queries. We therefore score the in-scope subset (gold keyword present) and
    # report negative-rejection on the out-of-scope rest separately.
    print(f"{'lang':5} {'n_train':>8} {'qry_n':>6} {'eval':>5} | "
          f"{'shipped_exact':>13} {'new_exact':>10} | {'shipped_f1':>10} {'new_f1':>7} | "
          f"{'neg_n':>5} {'sh_rej':>6} {'new_rej':>7}")
    for lang in args.langs.split(","):
        rows = load(lang)
        if len(rows) < 40:
            print(f"{lang:5}  (only {len(rows)} rows — skip)")
            continue
        gold = load_gold(lang)
        if gold:
            train, test, kind = rows, gold, "gold"
        else:  # fall back to an in-distribution hold-out
            rng.shuffle(rows)
            k = int(len(rows) * args.test_frac)
            train, test, kind = rows[k:], rows[:k], "split"
        pos = [t for t in test if t["keyword"]]
        neg = [t for t in test if not t["keyword"]]

        new_model = fit(lang, train)
        joblib.dump(new_model.model, os.path.join(OUT, f"kx_{lang}.pkl"))

        def reject(m):
            return sum(m.extract_keyword(t["text"]) == "" for t in neg) / len(neg) if neg else float("nan")

        new_scores = evaluate(new_model, pos)
        new_rej = reject(new_model)
        shipped_path = os.path.join(SHIPPED, f"kx_{lang}.pkl")
        if os.path.exists(shipped_path):
            shipped = SearchtermExtractorCRF(lang)
            shipped.load(shipped_path)
            sh = evaluate(shipped, pos)
            sh_rej = reject(shipped)
        else:
            sh = {"exact": float("nan"), "f1": float("nan")}
            sh_rej = float("nan")
        print(f"{lang:5} {len(train):8d} {len(pos):6d} {kind:>5} | "
              f"{sh['exact']:13.3f} {new_scores['exact']:10.3f} | {sh['f1']:10.3f} {new_scores['f1']:7.3f} | "
              f"{len(neg):5d} {sh_rej:6.3f} {new_rej:7.3f}")


if __name__ == "__main__":
    main()
