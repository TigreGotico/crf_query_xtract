# Training dataset

`train/build_dataset.py` builds a multilingual, token-level **search-term
extraction** dataset: for each utterance, every token carries a `B-KW` / `I-KW`
/ `O` label marking the span a user would send to a common-query, DuckDuckGo, or
music skill. This is what the CRF learns to predict.

## Schema

One JSON object per line in `train/data/<lang>.jsonl`:

| field | type | meaning |
| --- | --- | --- |
| `lang` | str | CRF language code (`ca da de en eu fr gl it pt`) |
| `text` | str | the utterance (whitespace-joined tokens) |
| `tokens` | list[str] | tokens, as produced by the Brill tokenizer |
| `pos` | list[str] | coarse POS tag per token (the CRF's main feature) |
| `labels` | list[str] | `B-KW` / `I-KW` / `O`, one per token |
| `source` | str | `slot_filling` \| `music` \| `common_query` |
| `keyword` | str | the gold search term (the labelled span) |

`train/data/stats.json` holds row counts per language and source.

## Sources

All three feed one span-labelling routine that locates the keyword **by token
position** (tokenise the utterance, tokenise the value, match the subsequence) —
not the case-sensitive set-membership the legacy synthesiser used, which dropped
the head of multi-word and possessive terms.

- **`slot_filling`** — OVOS locale `{query}` templates exported by
  [ovos-localize](https://github.com/OpenVoiceOS/ovos-localize) (common-query and
  DDG-solver skills). The `{query}` slot *is* the search term. `(a|b)`
  alternations are expanded with
  [ovos-spec-tools](https://github.com/OpenVoiceOS/ovos-spec-tools) `expand()`,
  the slot is filled with a real entity, and the inserted span is labelled.
  Templates carrying other (constrained) slots are skipped so no unfilled
  placeholder is ever labelled.
- **`music`** — [OpenVoiceOS/music_queries_templates](https://huggingface.co/datasets/OpenVoiceOS/music_queries_templates):
  `{artist_name}` / `{album_name}` / `{track_name}` slot templates filled with
  real music entities and span-labelled the same way.
- **`intents_eval`** — [OpenVoiceOS/intents-for-eval](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval)
  `<locale>-templates`: `{slot}` templates carrying **in-language slot
  examples**. Every slot is filled from its examples (so sentences are fully
  realised); only *content/entity* slots (`song`, `artist`, `place_name`,
  `query`, … — see `CONTENT_SLOTS`) are labelled KW. Constrained slots (time,
  date, number, volume) are filled but left `O`.
- **`massive`** — [OpenVoiceOS/massive-templates](https://huggingface.co/datasets/OpenVoiceOS/massive-templates):
  the MASSIVE corpus in the same template+examples shape across 50+ locales,
  labelled identically. Templates with no content slot become bounded all-`O`
  **negatives** that teach the model when *not* to extract.
- **`common_query`** — [OpenVoiceOS/ovos-common-query-intents](https://huggingface.co/datasets/OpenVoiceOS/ovos-common-query-intents):
  real, natural questions with no markup. The local Gemma server labels the
  search-term span (verbatim substring, validated); these terms are recycled as
  native fill values for `slot_filling`, so the synthetic utterances use
  language-appropriate entities.

## Languages

The 11 languages with both a Brill POS tagger (`brill_postaggers`) and data:
`ca da de en es eu fr gl it nl pt`. `es` and `nl` gain support here for the
first time (the shipped package previously claimed but never shipped them).

## Gold evaluation split

`train/data/gold/<lang>.jsonl` is the curated `<locale>-test` split of
intents-for-eval (filled utterance + `expected_slots`); the gold search term is
the content-slot value(s) in utterance order. This is the independent benchmark
`train/train_from_dataset.py` scores against — not a hold-out of the training
data.

## Entity pool

Per language the fill pool seeds from `train/keywords_<lang>.txt` and grows with
the Gemma-extracted spans from `common_query`. Multi-word entities are kept whole
so the model learns full spans (e.g. *speed of light*, not *speed*).

## Building

```bash
python train/build_dataset.py                 # all langs, all common-query
python train/build_dataset.py --langs en,pt   # subset
python train/build_dataset.py --no-gemma      # deterministic sources only (offline)
```

`slot_filling` and `music` are offline (seed pools ship in `train/`). The
`common_query` source needs the local Gemma server (`LLM_ENDPOINT` /
`LLM_MODEL`); `--no-gemma` skips it. The HF datasets download once into the
shared cache.

## Counts

50,982 training rows over 11 languages + a 4,400-row gold split (see
`train/data/stats.json`). Per-language totals range from ~840 (eu, gl — no MASSIVE
coverage) to ~8,600 (it); `massive` and `slot_filling` are capped at 4,000/lang.

## Retraining benchmark

`train/train_from_dataset.py` fits a fresh CRF per language and scores it against
the shipped model on the **gold split** (`train/data/gold/`). Because the
extractor runs behind an intent gate (it only sees utterances already classified
as search queries), the score that matters is over the **in-scope subset** — gold
rows that contain a search term (exact whole-keyword match / token F1):

| lang | shipped exact | new exact | shipped F1 | new F1 |
| --- | --- | --- | --- | --- |
| en | 0.34 | 1.00 | 0.58 | 1.00 |
| pt | 0.52 | 0.90 | 0.63 | 0.96 |
| fr | 0.26 | 0.94 | 0.47 | 0.98 |
| it | 0.04 | 0.91 | 0.49 | 0.97 |
| ca | 0.16 | 0.94 | 0.44 | 0.97 |
| de | 0.16 | 0.72 | 0.34 | 0.86 |
| da | 0.16 | 0.88 | 0.34 | 0.97 |
| eu | 0.00 | 0.59 | 0.30 | 0.79 |
| gl | 0.06 | 0.81 | 0.39 | 0.95 |
| es | – | 0.74 | – | 0.90 |
| nl | – | 0.94 | – | 0.97 |

Caveats: the in-scope subset is small (~32 rows/lang — the gold set is ~92%
out-of-scope smarthome/timer commands), so these are strong but limited samples.
Neither the shipped nor the new model rejects negatives well (returns `""` for
~5–15% of no-search-term utterances) because of the first-noun fallback; that only
matters if the extractor is used without an upstream intent classifier. Candidate
models land in `train/out/` and are **not** promoted over the shipped
`crf_query_xtract/kx_*.pkl` automatically.

## Publishing

The combined `train/data/*.jsonl` is a self-contained, HF-publishable
token-classification dataset (BIO search-term tagging). The `common_query` rows
are LLM silver labels and should be spot-checked before being treated as gold.
