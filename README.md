## Overview

🔎 The **Searchterm Extractor CRF** is a keyword extraction module designed for **OVOS (Open Voice OS)** common query skills, such as **Wikipedia** and **DuckDuckGo (DDG)**. This tool helps identify the most relevant search keywords from a user's spoken or typed query, enabling seamless integration with OVOS' search functionalities.

## How It Works 🚀

This extractor uses a **Conditional Random Field (CRF)** model to identify relevant keywords within input text. It operates in two stages: 
1. **POS Tagging**: Each word in the input text is tagged with **Part-of-Speech (POS)** information to better understand the sentence structure.
2. **Keyword Extraction**: The CRF model predicts which words are most likely to be relevant search terms based on surrounding context, using word features and POS tags.

By leveraging both **word features** (e.g., lowercase form) and **POS tags** (e.g., noun, verb), this system efficiently identifies important keywords, making it suitable for real-time applications.

### Conditional Random Fields (CRF) 🧠

**Conditional Random Fields (CRF)** are a class of probabilistic models used for **sequence labeling tasks** like **Named Entity Recognition (NER)** and **Part-of-Speech Tagging**. Unlike simpler models, CRFs evaluate the context of each word within a sentence to make predictions about its label. This context-awareness helps CRFs excel in tasks that require understanding of how words interact with their neighboring words, making them particularly effective for **keyword extraction**.

In the case of the **Searchterm Extractor CRF**, CRFs:
- Take into account the **POS tags** of neighboring words.
- Label each word as either a relevant **keyword** (`K`) or **non-keyword** (`O`).

### Brill POS Tagger 🏷️

The system utilizes the **Brill POS Tagger**, a rule-based part-of-speech tagger, to assign grammatical categories (e.g., noun, verb, adjective) to words. The Brill tagger applies **transformation-based learning**, a technique that iteratively refines POS tags by applying rules based on context. This method enhances the accuracy of POS tagging before the CRF model performs the final keyword extraction.

This combination of CRFs and Brill POS tagging provides a robust approach for **lightweight and efficient keyword extraction** without the need for deep learning, making it well-suited for real-time environments.

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

## Usage 🛠️

You can easily use the extractor to process search queries:

```python
from searchterm_extractor import SearchtermExtractorCRF

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

- **English** (`kx_en.pkl`)
- **German** (`kx_de.pkl`)
- **French** (`kx_fr.pkl`)
- **Italian** (`kx_it.pkl`)
- **Portuguese** (`kx_pt.pkl`)
- **Danish** (`kx_da.pkl`)
- **Spanish** (`kx_es.pkl`)
- **Dutch** (`kx_nl.pkl`)
- **Basque** (`kx_eu.pkl`)
- **Galician** (`kx_gl.pkl`)
- **Catalan** (`kx_ca.pkl`)

If your language is not supported, you will need to **train a new Brill POS tagger**. Pre-trained Brill POS taggers for several languages can be found in the **[brill_postaggers repository](https://github.com/TigreGotico/brill_postaggers)**.

If you encounter missing taggers, please open an issue in the main repository to request support for new languages.

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

