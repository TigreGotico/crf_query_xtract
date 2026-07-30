## Overview

`crf_query_xtract` extracts search keywords from a spoken or typed query. It is a
keyword extraction module for OVOS (Open Voice OS) common query skills, such as
Wikipedia and DuckDuckGo (DDG).

## How it works

The extractor uses a Conditional Random Field (CRF) to label each token of a
query as a keyword (`K`) or not (`O`). Contiguous `K` tokens join into the
returned search term. A regex tokenizer splits the text, and each token gets
cheap orthographic features: lowercased form, 2/3-character prefixes and
suffixes, word shape, casing and digit flags, and the neighbouring tokens. There
is no part-of-speech tagger and no deep learning, so the only runtime
dependencies are `joblib` and `sklearn_crfsuite`. This keeps the extractor small
and fast for on-device use.

### Conditional Random Fields (CRF)

A Conditional Random Field is a probabilistic model for sequence labeling. It
scores a label for each token using the token's features and the labels of its
neighbours. This is what lets the model keep multi-word terms (for example
*speed of light*) intact, instead of picking a single word.

### Example

Given the input:

```plaintext
"Who invented the telephone?"
```

the extractor identifies "telephone" as the key search term.

## Installation

Install `crf_query_xtract` with pip:

```bash
pip install crf_query_xtract
```

Per-language models download from the Hub
([`TigreGotico/crf-query-xtract`](https://huggingface.co/TigreGotico/crf-query-xtract))
on first use and cache locally. They are trained on the
[`TigreGotico/search-term-extraction`](https://huggingface.co/datasets/TigreGotico/search-term-extraction)
dataset. Point `from_pretrained(..., repo_id=...)` or the `CRF_QUERY_XTRACT_REPO`
env var at any Hub repo or a local directory to use your own models.

## Usage

Use the extractor to process search queries:

```python
from crf_query_xtract import SearchtermExtractorCRF

# Initialize the extractor for the desired language
kx = SearchtermExtractorCRF.from_pretrained("en")

# Example sentence
sentence = "What is the speed of light?"

# Extract keywords from the sentence
keywords = kx.extract_keyword(sentence)

# Print the extracted keywords
print("Extracted keywords:", keywords)
```

### Expected output

```plaintext
Extracted keywords: speed of light
```

## Language support

The pretrained models cover these languages:

- Catalan (`kx_ca.pkl`)
- Danish (`kx_da.pkl`)
- German (`kx_de.pkl`)
- English (`kx_en.pkl`)
- Spanish (`kx_es.pkl`)
- Basque (`kx_eu.pkl`)
- French (`kx_fr.pkl`)
- Galician (`kx_gl.pkl`)
- Italian (`kx_it.pkl`)
- Dutch (`kx_nl.pkl`)
- Portuguese (`kx_pt.pkl`)

To support another language, train a model on data for it. The features are
language-agnostic, so no POS tagger is needed. See [`docs/dataset.md`](docs/dataset.md)
and [`docs/advanced.md`](docs/advanced.md).

### Contributing to the dataset

The dataset and training code live in the `train` folder. A quick way to help
improve the model is to add more sentence templates to the dataset. Training
data is thin for some languages, and more templates improve performance there.

### Dataset and training files

- Sentence templates: in `sentences_*.txt` files for each language.
- Keywords: in `keywords_*.txt` files for each language.
- Pre-trained models: in `kx_*.pkl` files for each language.

Training runs from the code in `train.py`.

## Training

Train the CRF model on your own data with the `Trainer` class.

The `Trainer` class generates tagged sentences by combining keywords with
sentence templates. It uses the Brill POS tagger for POS tagging and labels
words as either keywords or non-keywords.

To train the model:

1. Prepare keyword and sentence datasets.
2. Generate tagged sentences.
3. Train the CRF model on the tagged sentences.

## Related projects

- [OpenVoiceOS/ovos-localize](https://github.com/OpenVoiceOS/ovos-localize) — the locale templates this extractor's training data is built from.
- [OpenVoiceOS/ovos-spec-tools](https://github.com/OpenVoiceOS/ovos-spec-tools) — expands the template alternations used to build training data.
- [TigreGotico/crf-query-xtract](https://huggingface.co/TigreGotico/crf-query-xtract) — the model repo on the Hugging Face Hub.
- [TigreGotico/search-term-extraction](https://huggingface.co/datasets/TigreGotico/search-term-extraction) — the training dataset on the Hugging Face Hub.

## License

MIT License
