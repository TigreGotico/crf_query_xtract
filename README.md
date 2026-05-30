## Overview

🔎 The **Searchterm Extractor CRF** is a keyword extraction module designed for **OVOS (Open Voice OS)** common query skills, such as **Wikipedia** and **DuckDuckGo (DDG)**. This tool helps identify the most relevant search keywords from a user's spoken or typed query, enabling seamless integration with OVOS' search functionalities.

## How It Works 🚀

This extractor uses a **Conditional Random Field (CRF)** to label each token of a
query as a keyword (`K`) or not (`O`); contiguous `K` tokens are joined into the
returned search term. A regex tokenizer splits the text and each token is described
by cheap **orthographic features** — lowercased form, 2/3-character prefixes and
suffixes, word shape, casing and digit flags, and the neighbouring tokens. There is
no part-of-speech tagger and no deep learning, so the only runtime dependencies are
`joblib` and `sklearn_crfsuite`, keeping it small and fast for on-device use.

### Conditional Random Fields (CRF) 🧠

**Conditional Random Fields** are probabilistic models for **sequence labeling** —
they score a label for each token using its features *and* the labels of its
neighbours, which is what lets the model keep multi-word terms (e.g.
*speed of light*) intact rather than picking a single word.

### Example ✨

Given the input:

```plaintext
"Who invented the telephone?"
```

The extractor identifies **"telephone"** as the key search term.

## Installation 📦

You can install the **Searchterm Extractor CRF** via pip:

```bash
pip install crf_query_xtract
```

Per-language models are downloaded from the Hub
([`TigreGotico/crf-query-xtract`](https://huggingface.co/TigreGotico/crf-query-xtract))
on first use and cached. They are trained on the
[`TigreGotico/search-term-extraction`](https://huggingface.co/datasets/TigreGotico/search-term-extraction)
dataset. Point `from_pretrained(..., repo_id=...)` or the `CRF_QUERY_XTRACT_REPO`
env var at any Hub repo or a local directory to use your own models.

## Usage 🛠️

You can easily use the extractor to process search queries:

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

### Expected Output

```plaintext
Extracted keywords: speed of light
```

## Language Support 🌍

Currently, the **Searchterm Extractor CRF** supports several languages. The pretrained models include:

- **Catalan** (`kx_ca.pkl`)
- **Danish** (`kx_da.pkl`)
- **German** (`kx_de.pkl`)
- **English** (`kx_en.pkl`)
- **Spanish** (`kx_es.pkl`)
- **Basque** (`kx_eu.pkl`)
- **French** (`kx_fr.pkl`)
- **Galician** (`kx_gl.pkl`)
- **Italian** (`kx_it.pkl`)
- **Dutch** (`kx_nl.pkl`)
- **Portuguese** (`kx_pt.pkl`)

To support another language, train a model on data for it — the features are
language-agnostic, so no POS tagger is needed. See [`docs/dataset.md`](docs/dataset.md)
and [`docs/advanced.md`](docs/advanced.md).

### Contributing to the Dataset ✍️

The **dataset** and **training code** are available in the `train` folder. If you’d like to help improve the model, a quick way to contribute is by adding more **sentence templates** to the dataset. Currently, we don’t have enough training data for all languages, so **more templates** will greatly improve performance.

### Dataset & Training Files

- **Sentence Templates**: Found in `sentences_*.txt` files for each language.
- **Keywords**: Found in `keywords_*.txt` files for each language.
- **Pre-trained models**: Stored in `kx_*.pkl` files for each language.

Training is done using the code in the `train.py` file.


## Training

The CRF model can be trained on your own data using a custom **Trainer** class. 

The **Trainer** class generates tagged sentences by combining keywords with sentence templates. It uses the **Brill POS Tagger** for POS tagging and labels words as either **keywords** or **non-keywords**.

Here’s an overview of how you can train the model:
1. **Prepare keyword and sentence datasets**.
2. **Generate tagged sentences**.
3. **Train the CRF model** using the tagged sentences.


## License 📜

MIT License

