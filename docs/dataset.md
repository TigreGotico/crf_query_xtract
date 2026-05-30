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
- **`common_query`** — [OpenVoiceOS/ovos-common-query-intents](https://huggingface.co/datasets/OpenVoiceOS/ovos-common-query-intents):
  real, natural questions with no markup. The local Gemma server labels the
  search-term span (verbatim substring, validated); these terms are recycled as
  native fill values for `slot_filling`, so the synthetic utterances use
  language-appropriate entities.

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

7,977 rows (see `train/data/stats.json`):

| lang | total | slot_filling | music | common_query (Gemma) |
| --- | --- | --- | --- | --- |
| ca | 1026 | 1026 | – | – |
| da | 274 | 264 | – | 10 |
| de | 297 | 282 | – | 15 |
| en | 1108 | 515 | 401 | 192 |
| eu | 291 | 246 | – | 45 |
| fr | 329 | 314 | – | 15 |
| gl | 321 | 310 | – | 11 |
| it | 4010 | 4000 | – | 10 |
| pt | 321 | 310 | – | 11 |

## Retraining benchmark

`train/train_from_dataset.py` fits a fresh CRF per language and scores it against
the shipped model on a held-out 15% split (exact whole-keyword match / token F1):

| lang | shipped exact | new exact | shipped F1 | new F1 |
| --- | --- | --- | --- | --- |
| en | 0.49 | 0.82 | 0.74 | 0.93 |
| pt | 0.75 | 0.79 | 0.87 | 0.94 |
| de | 0.73 | 0.91 | 0.94 | 0.98 |
| fr | 0.33 | 0.69 | 0.59 | 0.87 |
| it | 0.42 | 0.98 | 0.82 | 1.00 |
| ca | 0.76 | 0.97 | 0.94 | 0.99 |
| da | 0.63 | 0.90 | 0.77 | 0.94 |
| eu | 0.12 | 0.79 | 0.42 | 0.93 |
| gl | 0.67 | 0.90 | 0.86 | 0.96 |

Candidate models land in `train/out/` and are **not** promoted over the shipped
`crf_query_xtract/kx_*.pkl` automatically. The test split shares construction
with the training data and the `common_query` rows are LLM silver labels, so
these numbers measure fit to the target distribution, not an independent gold
set — a hand-labelled eval split is the recommended confirmation before promoting.

## Publishing

The combined `train/data/*.jsonl` is a self-contained, HF-publishable
token-classification dataset (BIO search-term tagging). The `common_query` rows
are silver labels (LLM-generated) and should be spot-checked before being treated
as gold; a held-out hand-labelled split per language is the recommended next step
for evaluation.
