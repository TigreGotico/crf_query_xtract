"""Tokenisation and CRF features for the search-term extractor.

Dependency-light and POS-free: the in-house `quebra_frases` regex tokeniser plus
cheap orthographic features (prefix/suffix/shape/case/digit) that an ablation
showed match or beat Brill POS tags. Both the model and the dataset builder import
these so tokenisation — and therefore label alignment — is identical at train and
inference time.
"""
from __future__ import annotations

import re
from typing import Dict, List

try:
    from quebra_frases import word_tokenize as _word_tokenize
except ImportError:  # defensive fallback — same word/punct split
    _RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    def _word_tokenize(text):
        return _RE.findall(text)


def tokenize(text: str) -> List[str]:
    """Split into word and punctuation tokens (in-house `quebra_frases`)."""
    return _word_tokenize(text or "")


def shape(word: str) -> str:
    return "".join("X" if c.isupper() else "x" if c.islower()
                   else "d" if c.isdigit() else "-" for c in word[:8])


def word2features(tokens: List[str], i: int) -> Dict[str, object]:
    w = tokens[i]
    f: Dict[str, object] = {
        "w": w.lower(),
        "pre2": w[:2].lower(), "pre3": w[:3].lower(),
        "suf2": w[-2:].lower(), "suf3": w[-3:].lower(),
        "title": w.istitle(), "upper": w.isupper(), "digit": w.isdigit(),
        "shape": shape(w),
    }
    for d in (-1, 1, -2, 2):
        j = i + d
        if 0 <= j < len(tokens):
            f[f"{d}:w"] = tokens[j].lower()
    if i == 0:
        f["BOS"] = True
    if i == len(tokens) - 1:
        f["EOS"] = True
    return f


def sent2features(tokens: List[str]) -> List[Dict[str, object]]:
    return [word2features(tokens, i) for i in range(len(tokens))]
