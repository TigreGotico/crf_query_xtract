---
license: apache-2.0
task_categories:
- token-classification
language:
- ca
- da
- de
- en
- es
- eu
- fr
- gl
- it
- nl
- pt
tags:
- keyword-extraction
- search-term-extraction
- query-understanding
- voice-assistant
- topic-extraction
- ovos
pretty_name: Multilingual Search-Term Extraction
size_categories:
- 10K<n<100K
---

# Multilingual Search-Term Extraction

Token-level labels marking, in a voice-assistant query, the **search term** — the
minimal topic string you would hand to a knowledge base or search engine. Given
*"what is the speed of light?"* the target is *"speed of light"*; given
*"set volume to fifty"* the target is nothing (there is no topic to look up).

The task is **not** document keyphrase extraction and **not** full intent/slot
NLU. It answers one question: *what do I search for?* — the input the OVOS
common-query / DuckDuckGo / Wikipedia skills send downstream.

## Task & label scheme

Sequence labeling with three tags per token:

| tag | meaning |
| --- | --- |
| `O` | not part of the search term |
| `B-KW` | first token of a search-term span |
| `I-KW` | continuation of a search-term span |

Contiguous `B-KW`/`I-KW` tokens form the search term; an utterance with no topic
(smart-home, timers, volume…) is all `O`. These **negatives are kept on purpose**
— the extractor runs behind an intent gate but should still not hallucinate a
topic where there is none.

## Schema

Each row: `lang`, `tokens` (list[str], `quebra_frases` tokenizer), `tags`
(`ClassLabel` sequence), `text`, `keyword` (the target string), `source`.

## Configs & splits

One **config per language** (`ca da de en es eu fr gl it nl pt`), each with:

- `train` — ~50k rows total (templated + synthetic; see *Provenance*).
- `test` — a curated gold split (~16k rows total) from human-authored eval sets.

```python
from datasets import load_dataset
ds = load_dataset("TigreGotico/search-term-extraction", "en")  # train + test
```

## Provenance

Derived from permissively licensed OpenVoiceOS resources and one local-LLM step.
The `source` field on every row records where it came from:

| source | what | label quality |
| --- | --- | --- |
| `slot_filling` | OVOS locale `{query}` templates (ovos-localize) | deterministic (slot span) |
| `intents_eval` | [intents-for-eval](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) templates (Apache-2.0) | deterministic (content slots) |
| `massive` | [massive-templates](https://huggingface.co/datasets/OpenVoiceOS/massive-templates), the MASSIVE corpus (Apache-2.0) | deterministic (content slots) |
| `music` | [music_queries_templates](https://huggingface.co/datasets/OpenVoiceOS/music_queries_templates) (MIT) | deterministic (slot span) |
| `ocp` | [OCP_templates](https://huggingface.co/datasets/OpenVoiceOS/OCP_templates) media query templates | deterministic (slot span) |
| `common_query` | real questions from [ovos-common-query-intents](https://huggingface.co/datasets/OpenVoiceOS/ovos-common-query-intents) | **silver** — span labelled by a local Gemma model, validated as a verbatim substring |
| `generated` | questions invented by a local Gemma model | **silver** — synthetic |

The `test` (gold) split is the `-test` configs of intents-for-eval and MASSIVE
(human-authored utterances with gold slot annotations).

Content slots are filled with real typed entities from
[Jarbas/WikidataMediaEntities](https://huggingface.co/datasets/Jarbas/WikidataMediaEntities)
(1.6M SFW entities across 53 types: artists, albums, movies, books, games,
people…), mapped to slot names; adult entity types are excluded.

## How this dataset was generated

The whole dataset is produced by `build_dataset.py` in the
[`crf_query_xtract`](https://github.com/TigreGotico/crf_query_xtract) repo
(`--dry-run`-able; deterministic given a seed except for the LLM steps). One
span-labelling routine is shared by every source: tokenise with `quebra_frases`,
locate the target value as a **token subsequence**, and tag that span `B-KW`/
`I-KW` — labels therefore always align to the published `tokens`.

1. **Template sources — deterministic, no LLM.** `slot_filling`, `intents_eval`,
   `massive` and `music` come from OVOS / MASSIVE `{slot}` templates. Slots are
   filled from each template's own example values; the *content* slot (the search
   term) is labelled, other slots are filled but left `O`, and `(a|b)`
   alternations are expanded with `ovos-spec-tools`. Templates whose only slots
   are constrained (time, volume…) become all-`O` negatives.
2. **`common_query` — local-LLM labelling.** Real questions from
   `ovos-common-query-intents` carry no markup, so a **locally-hosted Gemma model**
   (`ggml-org/gemma-4-26B-A4B-it`, run on the maintainer's own hardware) is asked
   for the search-term substring; a result is kept only if it is a **verbatim
   substring** of the question (otherwise dropped). Treat these as *silver*.
3. **`generated` — local-LLM synthesis.** The same Gemma model invents extra
   natural questions and their search term, kept under the same substring check.
   A small synthetic top-up, mainly for thin languages.
4. **Gold (`test`) split.** Taken verbatim from the human-authored `-test` configs
   of intents-for-eval and MASSIVE; the search term is the content-slot value(s)
   from those datasets' own gold annotations. No LLM labelling.

### Transparency on AI involvement

- **The construction pipeline, the labelling heuristics, and this card were
  written by [Anthropic's Claude](https://www.anthropic.com/claude) operating as
  an autonomous coding agent.** Claude wrote *code and documentation* — it did not
  author or label any row of data.
- **All in-dataset LLM labelling and synthesis (steps 2–3) were done by a
  local open-weights Gemma model, not by Claude.** Roughly 1.5% of `train` rows
  are LLM-touched (`common_query` + `generated`); the rest are template-derived.
- The gold split contains **no** model-generated labels.

## Quality, scope & limitations

- **Gold vs silver.** The `test` split is curated; in `train`, `common_query` and
  `generated` are LLM-labelled and the templated sources use a content-slot
  heuristic (a fixed all-list of entity slot names). Treat `train` as silver.
- **Synthetic distribution.** Most `train` rows are templates filled with
  entities; the real-query distribution differs. `common_query` and the gold
  split are the most natural.
- **What counts as the term** follows the source slot boundaries, so leading
  articles can be included (*"the speed of light"*). Multi-entity utterances
  concatenate spans in surface order.
- **Coverage** is uneven: `eu` and `gl` are thin (no MASSIVE coverage).

## Intended use

Train or evaluate a topic/search-term extractor that sits between intent
classification and a search/KB backend. Works for sequence-labeling models (CRF,
token classifiers) or as supervision for an LLM. The reference model trained on it
is [`crf_query_xtract`](https://github.com/TigreGotico/crf_query_xtract).

## License & attribution

Apache-2.0. Built from OpenVoiceOS datasets (Apache-2.0 / MIT) and the MASSIVE
corpus (Apache-2.0); please credit those upstreams alongside this dataset.
