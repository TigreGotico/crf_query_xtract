"""Build a multilingual BIO search-term dataset for the CRF extractor.

Produces token-level `B-KW`/`I-KW`/`O` sequences (with POS tags) where the
labelled span is the *search term* a user would send to a common-query / DDG /
music skill. Three sources, all sharing one span-labelling routine that tracks
the keyword by token position (not the lossy set-membership the old trainer
used):

  slot_filling : OVOS locale `{query}` templates (ovos-localize export). The
                 `{query}` slot IS the search term; `(a|b)` alternations are
                 expanded with ovos-spec-tools, then the slot is filled with a
                 real entity and the inserted span is labelled.
  music        : OpenVoiceOS/music_queries_templates — `{artist_name}` /
                 `{album_name}` / `{track_name}` slots filled the same way.
  common_query : OpenVoiceOS/ovos-common-query-intents — real natural questions
                 with no markup; the local Gemma server labels the search-term
                 span (augmentation). Extracted terms are recycled as fill
                 values for the slot_filling source, keeping entities native.

Output: train/data/<lang>.jsonl  (+ train/data/stats.json)
Each line: {lang, text, tokens, pos, labels, source, keyword}

Offline by default for slot_filling/music (seed entity pools ship in train/).
Gemma augmentation needs the local server; skip with --no-gemma.

Run::

    python train/build_dataset.py --langs en,pt          # subset
    CAP_TIME=1800 ../../tools/cap python train/build_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WS = os.path.abspath(os.path.join(REPO, "..", "..", ".."))
SLOT_FILL_DIR = os.path.join(WS, "ovos", "web", "ovos-localize", "data", "datasets", "slot_filling")
OUT_DIR = os.path.join(HERE, "data")

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://192.168.1.200:8000/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "ggml-org/gemma-4-26B-A4B-it-GGUF")

# CRF lang -> (slot_filling files, common-query lang code)
LANGS: Dict[str, Dict[str, object]] = {
    "ca": {"sf": ["ca-ES"], "cq": "ca"},
    "da": {"sf": ["da-DK"], "cq": "da"},
    "de": {"sf": ["de-DE"], "cq": "de"},
    "en": {"sf": ["en-US"], "cq": "en"},
    "eu": {"sf": ["eu-ES"], "cq": "eu"},
    "fr": {"sf": ["fr-FR"], "cq": "fr"},
    "gl": {"sf": ["gl-ES"], "cq": "gl"},
    "it": {"sf": ["it-IT"], "cq": "it"},
    "pt": {"sf": ["pt-PT", "pt-BR"], "cq": "pt"},
}

# Slots whose filler is a free search term (gets KW labels). Other slots are
# constrained values (brightness, offset, year...) — templates carrying them are
# skipped to avoid labelling unfilled placeholders.
SEARCH_SLOTS = {"query", "persona", "person", "entity", "search", "topic", "subject", "thing"}
MUSIC_SLOTS = {"artist_name", "album_name", "track_name", "song_name", "playlist_name",
               "genre_name", "artist", "album", "track", "song", "playlist", "genre"}

MUSIC_FILLS = [
    "The Beatles", "Miles Davis", "Daft Punk", "Amália Rodrigues", "Pink Floyd",
    "Beyoncé", "Kendrick Lamar", "Fela Kuti", "Caetano Veloso", "Radiohead",
    "Abbey Road", "Kind of Blue", "Random Access Memories", "Dark Side of the Moon",
    "To Pimp a Butterfly", "OK Computer", "Bitches Brew", "Lemonade",
    "Bohemian Rhapsody", "So What", "Get Lucky", "Comfortably Numb",
    "HUMBLE.", "Redemption Song", "Take Five", "Clair de Lune",
]

# ----------------------------------------------------------------------------
# tagging + span labelling
# ----------------------------------------------------------------------------
_TAGGERS: Dict[str, object] = {}


def tagger(lang: str):
    if lang not in _TAGGERS:
        from brill_postaggers import BrillPostagger
        _TAGGERS[lang] = BrillPostagger.from_pretrained(lang)
    return _TAGGERS[lang]


def _expander():
    try:
        from ovos_spec_tools import expand
    except ImportError:
        sys.path.insert(0, os.path.join(WS, "ovos", "tools", "ovos-spec-tools"))
        from ovos_spec_tools import expand
    return expand


def _find_subseq(seq: List[str], sub: List[str], near: int = 0) -> int:
    """Index of `sub` in `seq` (lowercased), preferring a match near `near`."""
    if not sub:
        return -1
    hits = [i for i in range(len(seq) - len(sub) + 1) if seq[i:i + len(sub)] == sub]
    if not hits:
        return -1
    return min(hits, key=lambda i: abs(i - near))


def label_span(lang: str, sentence: str, value: str, near_word: int = 0
               ) -> Optional[Tuple[List[str], List[str], List[str]]]:
    """Tokenise `sentence`, BIO-label the `value` token span. None if unmatched."""
    tg = tagger(lang)
    stoks = tg.tag(sentence)
    vtoks = tg.tag(value)
    sw = [w.lower() for w, _ in stoks]
    vw = [w.lower() for w, _ in vtoks]
    i = _find_subseq(sw, vw, near=near_word)
    if i < 0:
        return None
    words = [w for w, _ in stoks]
    pos = [p for _, p in stoks]
    labels = ["O"] * len(words)
    for j in range(i, i + len(vw)):
        labels[j] = "B-KW" if j == i else "I-KW"
    return words, pos, labels


def record(lang: str, text: str, value: str, source: str, near_word: int = 0
           ) -> Optional[dict]:
    out = label_span(lang, text, value, near_word=near_word)
    if out is None:
        return None
    words, pos, labels = out
    if "B-KW" not in labels:
        return None
    return {"lang": lang, "text": " ".join(words), "tokens": words, "pos": pos,
            "labels": labels, "source": source, "keyword": value}


# ----------------------------------------------------------------------------
# Gemma augmentation
# ----------------------------------------------------------------------------
def gemma_search_terms(sentences: List[str], lang: str, timeout: int = 120) -> List[Optional[str]]:
    """Label the search-term span in each sentence via the local Gemma server.

    Returns one term per sentence (None when the model declines or the term is
    not an exact substring). Batched into one request; parsed defensively.
    """
    import requests
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    prompt = (
        "You extract the SEARCH TERM from voice-assistant questions: the minimal "
        "noun phrase a user would type into Wikipedia or a search engine to answer "
        "the question (a person, place, work, event or concept). Keep it as a "
        "VERBATIM substring of the question — same words, same language, no "
        "rephrasing, no question words, no articles unless part of a name.\n"
        f"Language: {lang}.\n"
        "Return ONE JSON object mapping the item number (as string) to its search "
        "term, e.g. {\"0\": \"Pablo Picasso\"}. Use null if there is none.\n\n"
        f"{numbered}"
    )
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(LLM_ENDPOINT, json=body, timeout=timeout)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
    except Exception as e:  # noqa: BLE001 — degrade to no labels
        print(f"    gemma batch failed ({type(e).__name__}: {str(e)[:80]}) — skipping", flush=True)
        return [None] * len(sentences)
    out: List[Optional[str]] = []
    for i, s in enumerate(sentences):
        term = data.get(str(i))
        if isinstance(term, str) and term.strip() and term.strip().lower() in s.lower():
            out.append(term.strip())
        else:
            out.append(None)
    return out


# ----------------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------------
def seed_pool(lang: str) -> List[str]:
    path = os.path.join(HERE, f"keywords_{lang}.txt")
    if not os.path.exists(path):
        return []
    expand = _expander()
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:  # entities may contain literal parens, e.g. "(WHO)"
                out.extend(expand(line))
            except Exception:  # noqa: BLE001
                out.append(line)
    return list(dict.fromkeys(out))


def build_common_query(lang: str, cq_rows: List[str], pool: List[str],
                       budget: int, batch: int = 10) -> List[dict]:
    rows: List[dict] = []
    rows_in = cq_rows[:budget] if budget else cq_rows
    for k in range(0, len(rows_in), batch):
        chunk = rows_in[k:k + batch]
        terms = gemma_search_terms(chunk, lang)
        for sent, term in zip(chunk, terms):
            if not term:
                continue
            rec = record(lang, sent, term, "common_query")
            if rec:
                rows.append(rec)
                pool.append(term)
        print(f"    [{lang}] common_query {min(k + batch, len(rows_in))}/{len(rows_in)} "
              f"-> {len(rows)} labelled", flush=True)
    return rows


def build_slot_filling(lang: str, pool: List[str], cap: int, fills: int = 2) -> List[dict]:
    expand = _expander()
    rows: List[dict] = []
    seen = set()
    files = [os.path.join(SLOT_FILL_DIR, f"{c}.jsonl") for c in LANGS[lang]["sf"]]
    templates: List[Tuple[str, str]] = []  # (template, search_slot)
    for path in files:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                slots = set(obj.get("slots", []))
                target = slots & SEARCH_SLOTS
                if len(target) != 1:
                    continue
                templates.append((obj["template"], next(iter(target))))
    random.shuffle(templates)
    for template, slot in templates:
        if len(rows) >= cap:
            break
        try:
            variants = expand(template)
        except Exception:  # noqa: BLE001
            variants = [template]
        random.shuffle(variants)
        for variant in variants[:4]:
            # skip variants that still carry other (unfilled) slots
            tmp = variant.replace("{" + slot + "}", "\x00")
            if "{" in tmp and "}" in tmp:
                continue
            prefix = variant.split("{" + slot + "}")[0]
            near = len(prefix.split())
            for value in random.sample(pool, min(fills, len(pool))):
                text = variant.replace("{" + slot + "}", value)
                if "{" in text or "}" in text:  # other unfilled placeholder
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                rec = record(lang, text, value, "slot_filling", near_word=near)
                if rec:
                    rows.append(rec)
                if len(rows) >= cap:
                    break
    return rows


def build_music(music_templates: List[Tuple[str, str]], cap: int = 400) -> List[dict]:
    """music_queries_templates are English `{slot}` patterns."""
    rows: List[dict] = []
    seen = set()
    random.shuffle(music_templates)
    for category, template in music_templates:
        if len(rows) >= cap:
            break
        slots = [s for s in MUSIC_SLOTS if "{" + s + "}" in template]
        if len(slots) != 1:
            continue
        slot = slots[0]
        prefix = template.split("{" + slot + "}")[0]
        near = len(prefix.split())
        for value in random.sample(MUSIC_FILLS, 2):
            text = template.replace("{" + slot + "}", value)
            if "{" in text or "}" in text:  # other unfilled placeholder
                continue
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            rec = record("en", text, value, "music", near_word=near)
            if rec:
                rows.append(rec)
    return rows


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS), help="comma list of CRF langs")
    ap.add_argument("--slot-cap", type=int, default=4000, help="max slot_filling rows / lang")
    ap.add_argument("--gemma-budget", type=int, default=0,
                    help="max common-query sentences / lang to label (0 = all)")
    ap.add_argument("--no-gemma", action="store_true", help="skip Gemma augmentation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    langs = [l for l in args.langs.split(",") if l in LANGS]
    os.makedirs(OUT_DIR, exist_ok=True)

    # load HF sources once
    cq_by_lang: Dict[str, List[str]] = {}
    music_templates: List[Tuple[str, str]] = []
    if not args.no_gemma or True:
        from datasets import load_dataset
        cq = load_dataset("OpenVoiceOS/ovos-common-query-intents", split="train")
        for row in cq:
            cq_by_lang.setdefault(row["lang"], []).append(row["sentence"])
        music = load_dataset("OpenVoiceOS/music_queries_templates", split="train")
        music_templates = [(r["category"], r["template"]) for r in music]

    stats: Dict[str, Counter] = {}
    for lang in langs:
        print(f"== {lang} ==", flush=True)
        pool = seed_pool(lang)
        rows: List[dict] = []
        if not args.no_gemma:
            cq_rows = cq_by_lang.get(LANGS[lang]["cq"], [])
            if cq_rows:
                rows += build_common_query(lang, cq_rows, pool, args.gemma_budget)
        rows += build_slot_filling(lang, pool, args.slot_cap)
        if lang == "en":
            rows += build_music(music_templates)
        random.shuffle(rows)
        out_path = os.path.join(OUT_DIR, f"{lang}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter(r["source"] for r in rows)
        stats[lang] = c
        print(f"   wrote {len(rows)} rows -> {out_path}  ({dict(c)})", flush=True)

    summary = {lang: {"total": sum(c.values()), "by_source": dict(c)}
               for lang, c in stats.items()}
    summary["_meta"] = {
        "sources": ["slot_filling (ovos-localize)", "music_queries_templates (HF)",
                    "common_query gemma-labelled (HF)"],
        "label_scheme": "BIO (B-KW/I-KW/O)",
        "pos_tagger": "brill_postaggers",
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nTOTAL:", sum(s["total"] for k, s in summary.items() if k != "_meta"), "rows")
    print(json.dumps({k: v for k, v in summary.items() if k != "_meta"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
