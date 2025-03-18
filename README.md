######

## Overview

🔎 The **Searchterm Extractor CRF** is a keyword extraction module designed for **OVOS (Open Voice OS)** common query skills, such as Wikipedia and DuckDuckGo (DDG). It helps identify the most relevant search keywords from a user’s spoken or typed query.

## How It Works 🚀

This extractor uses a **Conditional Random Field (CRF)** model trained on keyword patterns and sentence structures. It processes input text by tagging each word with **Part-of-Speech (POS) information** and then predicts which words are most relevant as search terms. The model relies only on **word features and POS tags**, making it lightweight and efficient for real-time applications.

### Conditional Random Fields (CRF) 🧠

CRFs are a type of probabilistic model used for **sequence labeling** tasks, such as **Named Entity Recognition (NER)** and **Part-of-Speech Tagging**. Unlike simpler models, CRFs consider the context of surrounding words to make more accurate predictions, making them well-suited for extracting meaningful search terms from natural language queries.

### Brill POS Tagger 🏷️

The system uses a **Brill POS Tagger**, a rule-based part-of-speech tagger that assigns grammatical categories (e.g., noun, verb) to words based on a predefined rule set. This helps the model better understand sentence structure before extracting relevant search terms. Unlike statistical models, the Brill tagger applies transformation-based learning to iteratively refine POS labels.

### Example ✨

Given the input:

```plaintext
"Who invented the telephone?"
```

The extractor identifies **"telephone"** as the key search term.

## Installation 📦

```bash
pip install crf_query_xtract
```

## Usage 🛠️

```python
from searchterm_extractor import SearchtermExtractorCRF

kx = SearchtermExtractorCRF.from_pretrained("en")

sentence = "What is the speed of light?"
keywords = kx.extract_keyword(sentence)

print("Extracted keywords:", keywords)
```

### Expected Output

```plaintext
Extracted keywords: speed of light
```

## Notes 📝

- This extractor is specifically designed for **search queries** and works best with OVOS common query skills.
- It **does not** use deep learning or advanced semantics—only **word and POS-based features**.
- If no clear keyword is found, it defaults to returning the **first noun** in the query.
- Currently depends on pre-trained models from **[brill\_postaggers](https://github.com/TigreGotico/brill_postaggers)**. Future updates will make this configurable.

## License 📜

MIT License

