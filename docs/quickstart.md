# Quickstart — query to search term

`crf_query_xtract` turns a full-sentence question into the handful of words you
would actually type into a search box. "Who invented the telephone?" becomes
`telephone`. It is built for OVOS common-query skills (Wikipedia, DuckDuckGo)
that need a clean search term, not the whole utterance.

## 1. Install

```bash
pip install crf_query_xtract
```

Small runtime dependencies install with it: `joblib`, `sklearn_crfsuite`, the
in-house `quebra_frases` tokenizer, and `huggingface_hub`. The per-language CRF
model downloads from the Hub (from `TigreGotico/crf-query-xtract`) on first use
and caches locally. There is no POS tagger and no GPU requirement.

## 2. The one idea

A pretrained per-language CRF model labels each token as a keyword (`K`) or not
(`O`), then the contiguous `K` runs join into the search term. Tokens come from
the `quebra_frases` regex tokenizer and get cheap orthographic features
(prefixes, suffixes, word shape, casing). There is no POS tagger, no GPU, and no
deep net, so it runs in milliseconds.

Load a model with `from_pretrained` and call `extract_keyword`:

```python
from crf_query_xtract import SearchtermExtractorCRF

kx = SearchtermExtractorCRF.from_pretrained("en")
print(kx.extract_keyword("who invented the telephone"))   # telephone
```

`extract_keyword` always returns a `str`: the joined keyword(s), or `""` when
the model labels no token as a keyword.

## 3. First real call

Portuguese, the way a Lusophone user would phrase it:

```python
from crf_query_xtract import SearchtermExtractorCRF

kx = SearchtermExtractorCRF.from_pretrained("pt")

for q in ["quem inventou o telefone",
          "qual a velocidade da luz",
          "quem descobriu o fogo"]:
    print(q, "->", kx.extract_keyword(q))
# quem inventou o telefone   -> telefone
# qual a velocidade da luz   -> velocidade da luz
# quem descobriu o fogo      -> fogo
```

Multi-word terms come back as a single space-joined string (`velocidade da
luz`), because the `K` run spans several tokens.

## 4. Supported languages

Models are available for: `ca` `da` `de` `en` `es` `eu` `fr` `gl` `it` `nl` `pt`.
Pass a plain code or a locale. `from_pretrained` lowercases the code and drops
the region, so `"pt-BR"` loads the `pt` model:

```python
kx = SearchtermExtractorCRF.from_pretrained("pt-BR")          # same as "pt"
kx = SearchtermExtractorCRF.from_pretrained("en", repo_id="me/my-crf")  # your own models
```

`repo_id` (or the `CRF_QUERY_XTRACT_REPO` env var) points at any Hub repo or a
local directory of `kx_<lang>.pkl` files, so you can swap in your own models. To
train them, see [advanced.md](advanced.md).

---
[Home](../README.md) · [API reference →](api.md)
