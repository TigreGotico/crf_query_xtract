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
import ast
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

# Languages with both a brill POS tagger AND data. Per lang:
#   sf  : ovos-localize slot_filling locale file(s)
#   cq  : ovos-common-query-intents lang code
#   tpl : intents-for-eval / massive-templates locale (config prefix)
LANGS: Dict[str, Dict[str, object]] = {
    "ca": {"sf": ["ca-ES"], "cq": "ca", "tpl": "ca-ES"},
    "da": {"sf": ["da-DK"], "cq": "da", "tpl": "da-DK"},
    "de": {"sf": ["de-DE"], "cq": "de", "tpl": "de-DE"},
    "en": {"sf": ["en-US"], "cq": "en", "tpl": "en-US"},
    "es": {"sf": ["es-ES", "es-419"], "cq": "es", "tpl": "es-ES"},
    "eu": {"sf": ["eu-ES"], "cq": "eu", "tpl": "eu-ES"},
    "fr": {"sf": ["fr-FR"], "cq": "fr", "tpl": "fr-FR"},
    "gl": {"sf": ["gl-ES"], "cq": "gl", "tpl": "gl-ES"},
    "it": {"sf": ["it-IT"], "cq": "it", "tpl": "it-IT"},
    "nl": {"sf": ["nl-NL"], "cq": "nl", "tpl": "nl-NL"},
    "pt": {"sf": ["pt-PT", "pt-BR"], "cq": "pt", "tpl": "pt-PT"},
}

INTENTS_EVAL = "OpenVoiceOS/intents-for-eval"
MASSIVE = "OpenVoiceOS/massive-templates"

# Free-text content/entity slots whose filler is a search term (labelled KW).
# Everything else (time, date, number, volume, colour, ...) is filled but left O
# — useful negatives teaching the model when NOT to extract.
CONTENT_SLOTS = {
    "song", "song_name", "artist", "artist_name", "album", "album_name", "track",
    "track_name", "playlist", "playlist_name", "genre", "music_genre", "media_type",
    "podcast_name", "podcast_descriptor", "radio_name", "audiobook_name",
    "audiobook_author", "movie_name", "movie_type", "game_name", "app_name", "query",
    "person", "person_name", "place_name", "business_name", "business_type",
    "food_type", "drink_type", "ingredient", "news_topic", "definition_word",
    "transport_name", "event_name", "artist_or_band", "song_or_album",
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
# tokenisation + span labelling (shares the model's tokeniser so labels align)
# ----------------------------------------------------------------------------
from crf_query_xtract.features import tokenize


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


def label_span(sentence: str, value: str, near_word: int = 0
               ) -> Optional[Tuple[List[str], List[str]]]:
    """Tokenise `sentence`, BIO-label the `value` token span. None if unmatched."""
    words = tokenize(sentence)
    vw = [w.lower() for w in tokenize(value)]
    sw = [w.lower() for w in words]
    i = _find_subseq(sw, vw, near=near_word)
    if i < 0:
        return None
    labels = ["O"] * len(words)
    for j in range(i, i + len(vw)):
        labels[j] = "B-KW" if j == i else "I-KW"
    return words, labels


def record(lang: str, text: str, value: str, source: str, near_word: int = 0
           ) -> Optional[dict]:
    out = label_span(text, value, near_word=near_word)
    if out is None:
        return None
    words, labels = out
    if "B-KW" not in labels:
        return None
    return {"lang": lang, "text": " ".join(words), "tokens": words,
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


def gemma_generate(lang: str, n: int, timeout: int = 180) -> List[Tuple[str, str]]:
    """Ask Gemma to invent natural search questions + their term, in `lang`."""
    import requests
    prompt = (
        f"Generate {n} short, natural questions a user would ask a voice assistant, "
        f"in the language with code '{lang}'. Each must be answerable by looking up a "
        "single topic — a person, place, work, event, or scientific/general concept. "
        "Vary the phrasing and the topics. For each, also give the minimal search term: "
        "a VERBATIM substring of the question (no question words, no trailing articles). "
        "Return ONE JSON object: {\"items\": [{\"q\": <question>, \"kw\": <search term>}, ...]}."
    )
    body = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "response_format": {"type": "json_object"}}
    try:
        r = requests.post(LLM_ENDPOINT, json=body, timeout=timeout)
        r.raise_for_status()
        items = json.loads(r.json()["choices"][0]["message"]["content"]).get("items", [])
    except Exception as e:  # noqa: BLE001
        print(f"    [{lang}] gemma_generate failed ({type(e).__name__}) — skipping", flush=True)
        return []
    out = []
    for it in items:
        q, kw = str(it.get("q", "")).strip(), str(it.get("kw", "")).strip()
        if q and kw and kw.lower() in q.lower():
            out.append((q, kw))
    return out


def build_generated(lang: str, n: int, pool: List[str]) -> List[dict]:
    rows = []
    for q, kw in gemma_generate(lang, n):
        rec = record(lang, q, kw, "generated")
        if rec:
            rows.append(rec)
            pool.append(kw)
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
# intents-for-eval / massive-templates (slots carry in-language examples)
# ----------------------------------------------------------------------------
def _lit(s):
    if isinstance(s, (list, dict)):
        return s
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def label_multi(sentence: str, values: List[str]
                ) -> Optional[Tuple[List[str], List[str]]]:
    """BIO-label every `values` span in `sentence` (each its own B-KW...I-KW)."""
    words = tokenize(sentence)
    low = [w.lower() for w in words]
    labels = ["O"] * len(words)
    for value in values:
        vw = [w.lower() for w in tokenize(value)]
        if not vw:
            continue
        i = -1
        for start in range(len(low) - len(vw) + 1):
            if low[start:start + len(vw)] == vw and all(labels[start + k] == "O" for k in range(len(vw))):
                i = start
                break
        if i < 0:
            return None  # a content value did not land cleanly -> drop the row
        for k in range(len(vw)):
            labels[i + k] = "B-KW" if k == 0 else "I-KW"
    return words, labels


def realize(template: str, slots: List[dict], rng: random.Random
            ) -> Optional[Tuple[str, List[str]]]:
    """Fill every `{slot}` from its inline examples; return (text, content values)."""
    text = template
    content: List[str] = []
    for slot in slots or []:
        name = slot.get("name")
        examples = [e for e in (slot.get("examples") or []) if e and str(e).strip()]
        if not name or "{" + name + "}" not in text or not examples:
            return None
        value = str(rng.choice(examples)).strip()
        text = text.replace("{" + name + "}", value)
        if name in CONTENT_SLOTS:
            content.append(value)
    if "{" in text or "}" in text:
        return None
    return text, content


def build_templated(dataset: str, lang: str, cap: int, neg_frac: float,
                    rng: random.Random, fills: int = 2) -> List[dict]:
    from datasets import load_dataset
    cfg = f"{LANGS[lang]['tpl']}-templates"
    try:
        ds = load_dataset(dataset, cfg, split="train")
    except Exception as e:  # noqa: BLE001 — locale not in this dataset
        print(f"    [{lang}] {dataset.split('/')[-1]} {cfg}: unavailable ({type(e).__name__})", flush=True)
        return []
    rows: List[dict] = []
    seen = set()
    neg_cap = int(cap * neg_frac)
    neg = 0
    src = "intents_eval" if "intents-for-eval" in dataset else "massive"
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    for j in idx:
        if len(rows) >= cap:
            break
        row = ds[j]
        slots = _lit(row.get("slots")) or []
        for _ in range(fills):
            built = realize(row["template"], slots, rng)
            if built is None:
                continue
            text, content = built
            if not content:  # all-O negative — keep a bounded number
                if neg >= neg_cap:
                    continue
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            out = label_multi(text, content)
            if out is None:
                continue
            words, labels = out
            if not content and "B-KW" in labels:
                continue
            if content and "B-KW" not in labels:
                continue
            if not content:
                neg += 1
            rows.append({"lang": lang, "text": " ".join(words), "tokens": words,
                         "labels": labels, "source": src,
                         "keyword": " ".join(content)})
    return rows


def _gold_keyword(utt: str, slots: dict) -> str:
    """Content-slot values present in the utterance, in order of appearance."""
    pairs = [(utt.lower().find(str(v).lower()), str(v))
             for k, v in (slots or {}).items() if v and k in CONTENT_SLOTS
             and str(v).lower() in utt.lower()]
    pairs.sort()
    return " ".join(v for _, v in pairs)


def build_gold(lang: str, cap: int = 1500) -> int:
    """Curated gold eval from the `<locale>-test` splits of both template datasets."""
    from datasets import load_dataset
    gold_dir = os.path.join(OUT_DIR, "gold")
    os.makedirs(gold_dir, exist_ok=True)
    rows: List[dict] = []
    seen = set()
    for dataset, src in ((INTENTS_EVAL, "intents_eval"), (MASSIVE, "massive")):
        cfg = f"{LANGS[lang]['tpl']}-test"
        try:
            ds = load_dataset(dataset, cfg, split="test")
        except Exception:  # noqa: BLE001 — locale not in this dataset
            continue
        for row in ds:
            utt = row["utterance"]
            if utt.lower() in seen:
                continue
            seen.add(utt.lower())
            rows.append({"lang": lang, "text": utt,
                         "keyword": _gold_keyword(utt, _lit(row.get("expected_slots"))),
                         "intent": row.get("expected_intent"),
                         "domain": row.get("domain"), "source": src})
    rows = rows[:cap]
    with open(os.path.join(gold_dir, f"{lang}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS), help="comma list of CRF langs")
    ap.add_argument("--slot-cap", type=int, default=4000, help="max slot_filling rows / lang")
    ap.add_argument("--gemma-budget", type=int, default=0,
                    help="max common-query sentences / lang to label (0 = all)")
    ap.add_argument("--no-gemma", action="store_true", help="skip Gemma augmentation")
    ap.add_argument("--templated-cap", type=int, default=4000,
                    help="max rows / lang from EACH of intents-for-eval & massive-templates")
    ap.add_argument("--neg-frac", type=float, default=0.12,
                    help="fraction of templated rows that may be no-keyword negatives")
    ap.add_argument("--no-gold", action="store_true", help="skip the gold eval export")
    ap.add_argument("--gemma-generate", type=int, default=40,
                    help="synthetic search questions Gemma invents per lang (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)
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
            if args.gemma_generate:
                rows += build_generated(lang, args.gemma_generate, pool)
        rows += build_slot_filling(lang, pool, args.slot_cap)
        rows += build_templated(INTENTS_EVAL, lang, args.templated_cap, args.neg_frac, rng)
        rows += build_templated(MASSIVE, lang, args.templated_cap, args.neg_frac, rng)
        if lang == "en":
            rows += build_music(music_templates)
        random.shuffle(rows)
        out_path = os.path.join(OUT_DIR, f"{lang}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter(r["source"] for r in rows)
        stats[lang] = c
        ngold = 0 if args.no_gold else build_gold(lang)
        print(f"   wrote {len(rows)} rows -> {out_path}  ({dict(c)})  gold={ngold}", flush=True)

    summary = {lang: {"total": sum(c.values()), "by_source": dict(c)}
               for lang, c in stats.items()}
    summary["_meta"] = {
        "sources": ["slot_filling (ovos-localize)", "intents_eval (HF intents-for-eval)",
                    "massive (HF massive-templates)", "music_queries_templates (HF)",
                    "common_query gemma-labelled (HF)", "generated (gemma-invented)"],
        "label_scheme": "BIO (B-KW/I-KW/O)",
        "tokeniser": "crf_query_xtract.features.tokenize (regex, POS-free)",
        "gold_eval": "train/data/gold/<lang>.jsonl (intents-for-eval + massive test splits)",
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nTOTAL:", sum(s["total"] for k, s in summary.items() if k != "_meta"), "rows")
    print(json.dumps({k: v for k, v in summary.items() if k != "_meta"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
