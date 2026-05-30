# Advanced

Recipes, the OVOS plugin, and the sharp edges.

## Reuse one extractor per language

`from_pretrained` loads a CRF model from disk. In a loop or a service, build once
and keep it:

```python
from crf_query_xtract import SearchtermExtractorCRF

_CACHE = {}

def keyword(text: str, lang: str = "en") -> str:
    lang = lang.split("-")[0].lower()
    if lang not in _CACHE:
        _CACHE[lang] = SearchtermExtractorCRF.from_pretrained(lang)
    return _CACHE[lang].extract_keyword(text)

print(keyword("qual a capital de Portugal", "pt"))
```

## Feeding a search backend

The extractor's job is to hand a clean term to a search call. The keyword string
drops in directly:

```python
from urllib.parse import quote
from crf_query_xtract import SearchtermExtractorCRF

kx = SearchtermExtractorCRF.from_pretrained("en")
term = kx.extract_keyword("what is the speed of light")   # 'speed of light'
url = f"https://en.wikipedia.org/w/index.php?search={quote(term)}"
print(url)
```

An empty return (`""`) means the model labelled no keyword. Guard for it before
querying:

```python
term = kx.extract_keyword(text)
if not term:
    term = text          # fall back to the raw utterance
```

## The OVOS plugin

`crf_query_xtract` registers an OPM keyword-extractor plugin under the
`opm.keywords` entry-point group as `ovos-crf-brill-keyword-extractor`:

```python
from crf_query_xtract.opm import CRFBrillKeywordExtractor

plugin = CRFBrillKeywordExtractor()
plugin.supported_langs        # {'ca','da','de','en','es','eu','fr','gl','it','nl','pt'}
```

`extract(text, lang) -> Dict[str, float]` returns the keyword mapped to a
confidence of `1.0`, or `{}` when nothing is found:

```python
plugin.extract("who invented the telephone", "en")   # {'telephone': 1.0}
```

`get_extractor(lang)` caches one `from_pretrained` model per language.

## Training your own model

The models are trained from a token-classification dataset, not by hand. Two
scripts in `train/` drive it (see [dataset.md](dataset.md) for the full picture):

- `train/build_dataset.py` assembles `train/data/<lang>.jsonl` — `B-KW`/`I-KW`/`O`
  labelled tokens — plus a gold eval split.
- `train/train_from_dataset.py` fits a `sklearn_crfsuite.CRF` per language and
  writes candidates to `train/out/kx_<lang>.pkl`.

```bash
python train/build_dataset.py --langs pt
python train/train_from_dataset.py --langs pt   # -> train/out/kx_pt.pkl
```

Use what you trained — a single file, a local directory, or your own Hub repo:

```python
from crf_query_xtract import SearchtermExtractorCRF

# one file
kx = SearchtermExtractorCRF("pt"); kx.load("train/out/kx_pt.pkl")

# a directory of kx_<lang>.pkl, or a Hub repo
kx = SearchtermExtractorCRF.from_pretrained("pt", repo_id="train/out")
kx = SearchtermExtractorCRF.from_pretrained("pt", repo_id="me/my-crf")
```

Publish a model set to the Hub with `python train/push_model_to_hub.py --repo
me/my-crf`, or set `CRF_QUERY_XTRACT_REPO=me/my-crf` to make it the default.
Adding a language only needs data for it (the features are language-agnostic) —
no POS tagger to train.

## Gotchas

- **`from_pretrained` vs the bare constructor.** `SearchtermExtractorCRF(lang)`
  leaves `model = None`; calling `extract_keyword` on it raises
  `AttributeError: 'NoneType' object has no attribute 'predict'`. Use
  `from_pretrained` (or `load`) for extraction.
- **Region codes are stripped.** `from_pretrained("pt-BR")` loads the `pt` model;
  there is one model per base language, not per locale.
- **Keywords come back joined, not listed.** A multi-word term is one
  space-separated `str` (`"speed of light"`), not a list of tokens.
- **No negative rejection.** The model is meant to run behind an intent gate, so
  it returns the most keyword-like span it finds; on an utterance with no search
  term it may still return something. Gate on intent upstream.

## Where next

- [quickstart.md](quickstart.md) — install and first call
- [api.md](api.md) — signatures and return shapes
