"""Publish the search-term dataset to the Hugging Face Hub.

Builds a token-classification dataset (BIO `O`/`B-KW`/`I-KW`) from
`train/data/<lang>.jsonl` (train split) and `train/data/gold/<lang>.jsonl`
(`test` split), one config per language, and pushes it to the TigreGotico org.
The dataset card (`train/DATASET_CARD.md`) is uploaded as the repo README.

Needs an HF token with write access to the org (`huggingface-cli login`, or
`HF_TOKEN` in the environment). Nothing is uploaded under `--dry-run`.

Run::

    python train/push_to_hub.py --dry-run               # validate locally
    HF_TOKEN=... python train/push_to_hub.py            # publish
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Tuple

from datasets import (ClassLabel, Dataset, DatasetDict, Features, Sequence, Value)

from crf_query_xtract.features import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CARD = os.path.join(HERE, "DATASET_CARD.md")
LABELS = ["O", "B-KW", "I-KW"]
FEATURES = Features({
    "lang": Value("string"),
    "tokens": Sequence(Value("string")),
    "tags": Sequence(ClassLabel(names=LABELS)),
    "text": Value("string"),
    "keyword": Value("string"),
    "source": Value("string"),
})


def bio_from_keyword(text: str, keyword: str) -> Tuple[List[str], List[str]]:
    """Tokenise `text` and BIO-tag the `keyword` span (best effort)."""
    toks = tokenize(text)
    tags = ["O"] * len(toks)
    if not keyword:
        return toks, tags
    low = [t.lower() for t in toks]
    kw = [w.lower() for w in tokenize(keyword)]
    # contiguous match first
    for i in range(len(low) - len(kw) + 1):
        if low[i:i + len(kw)] == kw:
            for k in range(len(kw)):
                tags[i + k] = "B-KW" if k == 0 else "I-KW"
            return toks, tags
    # fall back: tag any token that belongs to the keyword (non-contiguous spans)
    kwset = set(kw)
    prev = False
    for i, w in enumerate(low):
        if w in kwset:
            tags[i] = "B-KW" if not prev else "I-KW"
            prev = True
        else:
            prev = False
    return toks, tags


def to_dataset(rows: List[dict], from_keyword: bool) -> Dataset:
    recs = []
    for r in rows:
        if from_keyword or "tokens" not in r or "labels" not in r:
            toks, tags = bio_from_keyword(r["text"], r.get("keyword", ""))
        else:
            toks, tags = r["tokens"], r["labels"]
        recs.append({
            "lang": r["lang"], "tokens": toks,
            "tags": [LABELS.index(t) for t in tags],
            "text": r["text"], "keyword": r.get("keyword", ""),
            "source": r.get("source", ""),
        })
    return Dataset.from_list(recs, features=FEATURES)


def load(path: str) -> List[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="TigreGotico/search-term-extraction")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    args = ap.parse_args()

    langs = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(DATA, "*.jsonl")))
    grand_train = grand_test = 0
    for lang in langs:
        train = to_dataset(load(os.path.join(DATA, f"{lang}.jsonl")), from_keyword=False)
        test = to_dataset(load(os.path.join(DATA, "gold", f"{lang}.jsonl")), from_keyword=True)
        dd = DatasetDict({"train": train, "test": test})
        grand_train += len(train)
        grand_test += len(test)
        print(f"  {lang}: train={len(train):6d}  test/gold={len(test):5d}")
        if not args.dry_run:
            dd.push_to_hub(args.repo, config_name=lang, private=args.private)
    print(f"TOTAL: {grand_train} train + {grand_test} gold over {len(langs)} languages")

    if args.dry_run:
        print("\n[dry-run] nothing uploaded. Sample (en train[0]):")
        ex = to_dataset(load(os.path.join(DATA, "en.jsonl"))[:1], from_keyword=False)[0]
        print("  ", {k: (v[:6] if isinstance(v, list) else v) for k, v in ex.items()})
        return

    if os.path.exists(CARD):
        # Merge the card body with an explicit configs index so it does not clobber
        # the per-language data_files metadata push_to_hub wrote.
        from huggingface_hub import DatasetCard
        card = DatasetCard.load(CARD)
        card.data["configs"] = [
            {"config_name": l, "data_files": [
                {"split": "train", "path": f"{l}/train-*"},
                {"split": "test", "path": f"{l}/test-*"}]}
            for l in langs]
        card.push_to_hub(args.repo, repo_type="dataset")
        print("uploaded dataset card (+ configs index)")
    print(f"published -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
