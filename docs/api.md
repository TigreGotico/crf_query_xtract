# API reference

The importable surface is one class plus a small features module. The OVOS
plugin wrapper lives in a submodule and is covered in [advanced.md](advanced.md).

```python
from crf_query_xtract import SearchtermExtractorCRF
```

## `SearchtermExtractorCRF`

A CRF keyword extractor. Hold one instance per language and reuse it. Loading a
model is the only setup.

### `SearchtermExtractorCRF(lang: str = None)`

Bare constructor. Stores `lang` and leaves `model = None`. On its own this
instance cannot extract. Call `load(...)`, or use the `from_pretrained`
classmethod, which does both. Use the bare constructor only when you are about
to train (assign a fitted `sklearn_crfsuite.CRF` to `.model`).

| Attribute | Type | Meaning |
| --- | --- | --- |
| `lang` | `str` | The language code passed in. |
| `model` | `CRF` or `None` | The loaded `sklearn_crfsuite.CRF`. `None` until loaded or trained. |

### `SearchtermExtractorCRF.from_pretrained(lang, repo_id=None) -> SearchtermExtractorCRF`

The normal entry point. Normalizes `lang` (`lang.split("-")[0].lower()`), so
`"PT"`, `"pt"` and `"pt-BR"` all resolve to `pt`, then loads `kx_<lang>.pkl`.

The model downloads from the Hub and caches: by default from `DEFAULT_REPO`
(`TigreGotico/crf-query-xtract`), overridable with the `CRF_QUERY_XTRACT_REPO`
env var or the `repo_id` argument. `repo_id` may be another Hub repo or a local
directory of `kx_<lang>.pkl` files. Pass your own to use your own models.

```python
kx = SearchtermExtractorCRF.from_pretrained("it")                       # bundled repo
kx = SearchtermExtractorCRF.from_pretrained("it", repo_id="me/my-crf")  # your Hub repo
kx = SearchtermExtractorCRF.from_pretrained("it", repo_id="/tmp/models")  # local dir
```

Models exist for `ca` `da` `de` `en` `es` `eu` `fr` `gl` `it` `nl` `pt`.
Requesting another language raises from the download (`hf_hub_download`).

### `extract_keyword(text: str) -> str`

The method you call. Steps:

1. Tokenize `text` with `quebra_frases` (regex word/punctuation split).
2. Build orthographic features per token (lowercased form, 2/3-char prefixes and
   suffixes, word shape, title/upper/digit flags, the neighbouring +/-2 tokens,
   `BOS`/`EOS` at the edges). See `crf_query_xtract.features`.
3. Predict a `K`/`O` label per token with the CRF.
4. Join each contiguous run of `K` tokens with spaces; join multiple runs with
   spaces too. If no token is labelled `K`, return `""`.

```python
kx = SearchtermExtractorCRF.from_pretrained("en")
kx.extract_keyword("what is the speed of light")   # 'the speed of light'
kx.extract_keyword("who discovered fire")          # 'fire'
```

The return is always a single `str`. A multi-token keyword is space-joined into
that one string. It is not a list.

### `load(path: str) -> None`

Loads a CRF model with `joblib.load(path)` into `self.model`. `from_pretrained`
calls this for you. Call it directly only to load a model you trained yourself.

```python
kx = SearchtermExtractorCRF("pt")
kx.load("/path/to/kx_pt.pkl")
kx.extract_keyword("quem inventou o telefone")     # 'telefone'
```

## `crf_query_xtract.features`

POS-free tokenisation and features, imported by both the model and the dataset
builder so training and inference use identical tokenisation.

- `tokenize(text: str) -> List[str]` — the `quebra_frases` word tokenizer (regex
  fallback if `quebra_frases` is unavailable).
- `word2features(tokens, i) -> dict` / `sent2features(tokens) -> List[dict]` —
  the orthographic feature dicts the CRF consumes. They take a token list, not
  `(word, pos)` tuples.

---
[← Quickstart](quickstart.md) · [Home](../README.md) · [Advanced →](advanced.md)
