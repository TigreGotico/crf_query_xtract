# Quickstart — query to search term

`crf_query_xtract` turns a full-sentence question into the handful of words you'd
actually type into a search box. "Who invented the telephone?" becomes
`telephone`. It is built for OVOS common-query skills (Wikipedia, DuckDuckGo)
that need a clean search term, not the whole utterance.

## 1. Install

```bash
pip install crf_query_xtract
```

Three small runtime deps install with it: `joblib`, `sklearn_crfsuite`, and the
in-house `quebra_frases` tokenizer. No model download, no POS tagger — the
per-language CRF models ship inside the package.

## 2. The one idea

A pretrained per-language **CRF** model labels each token as a keyword (`K`) or
not (`O`), then the contiguous `K` runs are joined into the search term. Tokens
come from the `quebra_frases` regex tokenizer and are described by cheap
orthographic features (prefixes, suffixes, word shape, casing) — no POS tagger,
no GPU, no deep net. It runs in milliseconds.

You load a model with `from_pretrained` and call `extract_keyword`:

```python
from crf_query_xtract import SearchtermExtractorCRF

kx = SearchtermExtractorCRF.from_pretrained("en")
print(kx.extract_keyword("who invented the telephone"))   # telephone
```

`extract_keyword` always returns a `str` — the joined keyword(s), or `""` when the
model labels no token as a keyword.

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

Multi-word terms come back as a single space-joined string (`velocidade da luz`),
because the `K` run spans several tokens.

## 4. Supported languages

Pretrained models ship for: `ca` `da` `de` `en` `es` `eu` `fr` `gl` `it` `nl` `pt`. Pass a
plain code or a locale — `from_pretrained` lowercases and drops the region, so
`"pt-BR"` loads the `pt` model:

```python
kx = SearchtermExtractorCRF.from_pretrained("pt-BR")   # same as "pt"
```

A language outside that set raises `FileNotFoundError` from the model load. To
add one, train your own model — see [advanced.md](advanced.md).

## Where next

- [api.md](api.md) — every public class, method, signature and return shape
- [advanced.md](advanced.md) — the OPM plugin, training your own model, gotchas
