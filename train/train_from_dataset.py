"""Train the CRF from the BIO dataset and score it on the gold split.

Reads `train/data/<lang>.jsonl`, collapses BIO -> K/O (the extractor joins
contiguous `K` spans, so K/O is enough), fits a CRF per language, and reports
exact-match + token F1 on the in-scope gold subset (utterances that contain a
search term — the only thing the extractor sees behind an intent gate), plus the
negative-rejection rate on the rest.

Models are written to `train/out/kx_<lang>.pkl` for review before they replace
`crf_query_xtract/kx_*.pkl`.

Run::

    python train/train_from_dataset.py --langs en,pt
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import joblib
from sklearn_crfsuite import CRF

from crf_query_xtract import SearchtermExtractorCRF
from crf_query_xtract.features import sent2features

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
ALL_LANGS = "ca,da,de,en,es,eu,fr,gl,it,nl,pt"


def load(lang: str) -> List[dict]:
    p = os.path.join(DATA, f"{lang}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def load_gold(lang: str) -> List[dict]:
    p = os.path.join(DATA, "gold", f"{lang}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def evaluate(model: SearchtermExtractorCRF, rows: List[dict]) -> Dict[str, float]:
    exact = tp = fp = fn = 0
    for r in rows:
        gold = set(r["keyword"].lower().split())
        pred = model.extract_keyword(r["text"])
        pred_set = set(pred.lower().split())
        exact += pred.lower() == r["keyword"].lower()
        tp += len(gold & pred_set)
        fp += len(pred_set - gold)
        fn += len(gold - pred_set)
    n = len(rows) or 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"exact": exact / n, "f1": f1}


def fit(lang: str, rows: List[dict]) -> SearchtermExtractorCRF:
    X = [sent2features(r["tokens"]) for r in rows]
    y = [["K" if l != "O" else "O" for l in r["labels"]] for r in rows]
    crf = CRF(algorithm="lbfgs", max_iterations=200, c1=0.1, c2=0.1, all_possible_transitions=True)
    crf.fit(X, y)
    model = SearchtermExtractorCRF(lang)
    model.model = crf
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=ALL_LANGS)
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="held-out fraction used only when no gold split exists")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import random
    rng = random.Random(args.seed)

    print(f"{'lang':5} {'n_train':>8} {'qry_n':>6} | {'exact':>6} {'f1':>6} | {'neg_n':>5} {'neg_rej':>7}")
    for lang in args.langs.split(","):
        rows = load(lang)
        if len(rows) < 40:
            print(f"{lang:5}  (only {len(rows)} rows — skip)")
            continue
        gold = load_gold(lang)
        if gold:
            train, test = rows, gold
        else:
            rng.shuffle(rows)
            k = int(len(rows) * args.test_frac)
            train, test = rows[k:], rows[:k]
        pos = [t for t in test if t["keyword"]]
        neg = [t for t in test if not t["keyword"]]

        model = fit(lang, train)
        joblib.dump(model.model, os.path.join(OUT, f"kx_{lang}.pkl"))

        s = evaluate(model, pos)
        neg_rej = (sum(model.extract_keyword(t["text"]) == "" for t in neg) / len(neg)
                   if neg else float("nan"))
        print(f"{lang:5} {len(train):8d} {len(pos):6d} | {s['exact']:6.3f} {s['f1']:6.3f} | "
              f"{len(neg):5d} {neg_rej:7.3f}")


if __name__ == "__main__":
    main()
