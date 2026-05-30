"""Dataset exploration plots -> docs/img/*.png.

Run::

    python train/plots.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IMG = os.path.join(os.path.dirname(HERE), "docs", "img")

SOURCES = ["slot_filling", "intents_eval", "massive", "music", "common_query", "generated"]
COLORS = {"slot_filling": "#4C72B0", "intents_eval": "#55A868", "massive": "#C44E52",
          "music": "#8172B3", "common_query": "#CCB974", "generated": "#DA8BC3"}


def load_all():
    by_lang = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(DATA, "*.jsonl"))):
        lang = os.path.basename(f)[:-6]
        by_lang[lang] = [json.loads(l) for l in open(f, encoding="utf-8")]
    return by_lang


def gold_counts():
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "gold", "*.jsonl"))):
        lang = os.path.basename(f)[:-6]
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        out[lang] = (sum(1 for r in rows if r["keyword"]), sum(1 for r in rows if not r["keyword"]))
    return out


def main():
    os.makedirs(IMG, exist_ok=True)
    by_lang = load_all()
    langs = sorted(by_lang, key=lambda l: -len(by_lang[l]))

    # 1) rows per language, stacked by source
    per = {l: Counter(r["source"] for r in by_lang[l]) for l in langs}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = [0] * len(langs)
    for s in SOURCES:
        vals = [per[l].get(s, 0) for l in langs]
        ax.bar(langs, vals, bottom=bottom, label=s, color=COLORS[s])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("Training rows per language, by source")
    ax.set_ylabel("rows"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "rows_by_lang_source.png"), dpi=110); plt.close(fig)

    # 2) keyword token-length distribution
    kw = Counter()
    for l in langs:
        for r in by_lang[l]:
            kw[len(r["keyword"].split())] += 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(k for k in kw if k <= 8)
    ax.bar([str(k) for k in ks], [kw[k] for k in ks], color="#4C72B0")
    ax.bar("9+", sum(v for k, v in kw.items() if k > 8), color="#C44E52")
    ax.set_title("Keyword length (tokens); 0 = negative / no search term")
    ax.set_xlabel("tokens in keyword"); ax.set_ylabel("rows")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "keyword_length.png"), dpi=110); plt.close(fig)

    # 3) token label distribution
    lab = Counter()
    for l in langs:
        for r in by_lang[l]:
            lab.update(r["labels"])
    fig, ax = plt.subplots(figsize=(5, 4))
    order = ["O", "B-KW", "I-KW"]
    ax.bar(order, [lab.get(k, 0) for k in order], color=["#bbbbbb", "#55A868", "#357d4a"])
    ax.set_title("Token label distribution"); ax.set_ylabel("tokens")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "label_distribution.png"), dpi=110); plt.close(fig)

    # 4) gold split: in-scope vs out-of-scope per language
    gc = gold_counts()
    gl = sorted(gc)
    fig, ax = plt.subplots(figsize=(9, 4))
    pos = [gc[l][0] for l in gl]; neg = [gc[l][1] for l in gl]
    ax.bar(gl, pos, label="has search term (in-scope)", color="#55A868")
    ax.bar(gl, neg, bottom=pos, label="no search term (out-of-scope)", color="#bbbbbb")
    ax.set_title("Gold eval split composition per language")
    ax.set_ylabel("rows"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "gold_split.png"), dpi=110); plt.close(fig)

    print("wrote 4 plots to", IMG)
    for l in langs:
        print(f"  {l}: {len(by_lang[l])} rows  gold in-scope={gc.get(l, ('?',))[0]}")


if __name__ == "__main__":
    main()
