# API reference

The importable surface is one class. The OVOS plugin wrapper lives in a
submodule and is covered in [advanced.md](advanced.md).

```python
from crf_query_xtract import SearchtermExtractorCRF
```

## `SearchtermExtractorCRF`

A CRF keyword extractor backed by a Brill POS tagger. Hold one instance per
language and reuse it — construction loads a tagger and (via `from_pretrained`)
a model from disk.

### `SearchtermExtractorCRF(lang: str)`

Bare constructor. Loads the Brill POS tagger for `lang` via
`BrillPostagger.from_pretrained(lang)` and leaves `model = None`. On its own this
instance **cannot extract** — you must also call `load(...)`, or use the
`from_pretrained` classmethod which does both. Use the bare constructor only when
you are about to train (see the `Trainer` subclass in `train/train.py`).

| Attribute | Type | Meaning |
| --- | --- | --- |
| `lang` | `str` | The language code passed in. |
| `tagger` | `BrillPostagger` | The POS tagger used before CRF prediction. |
| `model` | `CRF` or `None` | The loaded `sklearn_crfsuite.CRF`. `None` until loaded/trained. |
| `_keywords` | `List[str]` | Empty unless populated by a trainer. |
| `_dataset` | `List[str]` | Empty unless populated by a trainer. |

### `SearchtermExtractorCRF.from_pretrained(lang: str) -> SearchtermExtractorCRF`

The normal entry point. Normalizes `lang` (`lang.split("-")[0].lower()`), so
`"PT"`, `"pt"` and `"pt-BR"` all resolve to `pt`, then constructs the extractor
and loads the bundled `kx_<lang>.pkl`. Returns a ready-to-use instance.

```python
kx = SearchtermExtractorCRF.from_pretrained("it")
kx.extract_keyword("chi ha inventato il telefono")   # 'telefono'
```

Raises `FileNotFoundError` if no model ships for the resolved language. Shipped
models: `ca` `da` `de` `en` `eu` `fr` `gl` `it` `pt`.

### `extract_keyword(text: str) -> str`

The method you call. Steps:

1. POS-tag `text` with the Brill tagger → list of `(word, pos)` tuples.
2. Build windowed features per token (current ±2 tokens, each with word and POS;
   `BOS`/`EOS` flags at the edges).
3. Predict a `K`/`O` label per token with the CRF.
4. Join each contiguous run of `K` tokens with spaces; multiple runs are joined
   with spaces too.
5. **Fallback:** if no token is labelled `K`, return the first token tagged
   `NOUN`. If there is no noun either, return `""`.

```python
kx = SearchtermExtractorCRF.from_pretrained("en")
kx.extract_keyword("what is the speed of light")   # 'speed of light'
kx.extract_keyword("who discovered fire")          # 'fire'
```

Return is always a single `str`. A multi-token keyword is space-joined into that
one string; it is not a list.

### `load(path: str) -> None`

Loads a CRF model with `joblib.load(path)` into `self.model`. `from_pretrained`
calls this for you; call it directly only to load a model you trained yourself.

```python
kx = SearchtermExtractorCRF("pt")
kx.load("/path/to/kx_pt.pkl")
kx.extract_keyword("quem inventou o telefone")     # 'telefone'
```

## Feature helpers (internal)

`_word2features(sent, idx) -> dict` and `_sent2features(sent) -> List[dict]` build
the feature dicts the CRF consumes. They take `(word, pos)` tuples, not raw text.
They are public on the instance but exist for training and prediction internals —
`extract_keyword` is the supported call. The same feature builder is reused by
the trainer so a model and its runtime see identical features.

## Where next

- [quickstart.md](quickstart.md) — install and first call
- [advanced.md](advanced.md) — OPM plugin, training, gotchas
