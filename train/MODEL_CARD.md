---
license: apache-2.0
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
pipeline_tag: token-classification
tags:
- keyword-extraction
- search-term-extraction
- crf
- query-understanding
- voice-assistant
- ovos
datasets:
- TigreGotico/search-term-extraction
library_name: crf_query_xtract
---

# crf-query-xtract — multilingual search-term extractor

Per-language CRF models that label, in a voice-assistant query, the **search term**
— the minimal topic string to hand to a knowledge base or search engine. Given
*"what is the speed of light?"* → *"the speed of light"*; a command with no topic
(*"set volume to fifty"*) → `""`.

One `kx_<lang>.pkl` per language for `ca da de en es eu fr gl it nl pt`. Load them
through the [`crf_query_xtract`](https://github.com/TigreGotico/crf_query_xtract)
package, which downloads from this repo on first use.

## How to use

```bash
pip install crf_query_xtract
```

```python
from crf_query_xtract import SearchtermExtractorCRF

kx = SearchtermExtractorCRF.from_pretrained("en")   # downloads kx_en.pkl from here, cached
kx.extract_keyword("what is the speed of light")    # 'the speed of light'

# your own / a fork:
kx = SearchtermExtractorCRF.from_pretrained("en", repo_id="me/my-models")
kx = SearchtermExtractorCRF.from_pretrained("en", repo_id="/path/to/local/dir")
```

## Architecture

A single `sklearn_crfsuite.CRF` per language — **no POS tagger, no deep learning**.
Tokenise with the `quebra_frases` regex tokenizer; describe each token with cheap
orthographic features (lowercased form, 2/3-char prefixes/suffixes, word shape,
case/digit flags, the ±2 neighbour tokens, BOS/EOS); predict `O`/`B-KW`/`I-KW` and
join contiguous keyword tokens. CPU-friendly, millisecond inference. An ablation
found Brill POS features add no accuracy, so they were dropped.

## Evaluation

Scored on the gold split of the
[training dataset](https://huggingface.co/datasets/TigreGotico/search-term-extraction).
The extractor runs behind an intent gate, so the headline is the **in-scope**
subset (utterances that contain a search term): exact whole-keyword match / token
F1, plus negative-rejection on the rest.

| lang | exact | F1 | neg-reject |
| --- | --- | --- | --- |
| ca | 0.81 | 0.91 | 0.88 |
| da | 0.82 | 0.93 | 0.90 |
| de | 0.76 | 0.90 | 0.88 |
| en | 0.77 | 0.90 | 0.86 |
| es | 0.76 | 0.91 | 0.85 |
| eu | 0.48 | 0.71 | 0.99 |
| fr | 0.81 | 0.92 | 0.89 |
| gl | 0.83 | 0.95 | 0.97 |
| it | 0.76 | 0.89 | 0.83 |
| nl | 0.82 | 0.93 | 0.89 |
| pt | 0.79 | 0.90 | 0.89 |

`eu` is the weak spot (thin training data). The model has no forced fallback, so it
returns `""` when it labels no keyword.

## Intended use & limitations

Built to sit between intent classification and a search/KB backend in OVOS
common-query / DuckDuckGo / Wikipedia skills. It assumes its input is already a
search query (it is not an intent classifier). Trained largely on templated +
silver (LLM-labelled) data — see the dataset card for provenance and caveats.

## Training

Reproducible from the dataset with `train/train_from_dataset.py` in the package
repo. Training data: [TigreGotico/search-term-extraction](https://huggingface.co/datasets/TigreGotico/search-term-extraction)
(Apache-2.0).
