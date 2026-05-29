# Advanced

Recipes, the OVOS plugin, and the sharp edges.

## Reuse one extractor per language

Constructing an extractor loads a Brill tagger and a CRF model from disk. In a
loop or a service, build once and keep it:

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

An empty return (`""`) means the model found no keyword and the sentence had no
noun. Guard for it before querying:

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
plugin.supported_langs        # {'ca','da','en','eu','fr','gl','it','pt'}
```

`extract(text, lang) -> Dict[str, float]` returns the keyword mapped to a
confidence of `1.0`, or `{}` when nothing is found:

```python
plugin.extract("who invented the telephone", "en")   # {'telephone': 1.0}
```

The plugin's extractor cache (`get_extractor`) builds models with the bare
`SearchtermExtractorCRF(lang)` constructor, which does not load a `.pkl`. For a
guaranteed-loaded model, the direct path is the most robust:

```python
from crf_query_xtract import SearchtermExtractorCRF
kx = SearchtermExtractorCRF.from_pretrained("en")
{kx.extract_keyword("who invented the telephone"): 1.0}   # {'telephone': 1.0}
```

`supported_langs` on the plugin lists `ca da en eu fr gl it pt`; the bundled
models also include `de`, reachable through `from_pretrained("de")`.

## Training your own model

The training subclass lives in `train/train.py` as `Trainer`, which extends
`SearchtermExtractorCRF`. It builds tagged sentences by dropping keywords into
templates, then fits a `sklearn_crfsuite.CRF`.

Data lives in `train/` as two files per language:

- `keywords_<lang>.txt` — one keyword (or bracket-expansion template) per line.
- `sentences_<lang>.txt` — question templates with a `{keyword}` slot, e.g.
  `who invented {keyword}`.

Both files are expanded with `ovos_utils.bracket_expansion.expand_template`, so a
line like `(who|what) is {keyword}` fans out into multiple sentences.

```python
# run from inside the train/ directory — it reads files from the CWD
from train import Trainer

t = Trainer("pt")
t.train()                 # loads data, generates tagged sentences, fits the CRF
t.save("kx_pt.pkl")
```

Load and use what you trained through the same class:

```python
from crf_query_xtract import SearchtermExtractorCRF

kx = SearchtermExtractorCRF("pt")
kx.load("kx_pt.pkl")
kx.extract_keyword("quem inventou o telefone")
```

To add a language with no shipped model, you also need its Brill POS tagger.
Pretrained taggers for several languages come from
[brill_postaggers](https://github.com/TigreGotico/brill_postaggers); if yours is
missing, train a tagger there first.

## Gotchas

- **`from_pretrained` vs the bare constructor.** `SearchtermExtractorCRF(lang)`
  leaves `model = None`; calling `extract_keyword` on it raises
  `AttributeError: 'NoneType' object has no attribute 'predict'`. Use
  `from_pretrained` (or `load`) for extraction.
- **Region codes are stripped.** `from_pretrained("pt-BR")` loads the `pt` model;
  there is one model per base language, not per locale.
- **Keywords come back joined, not listed.** A multi-word term is one
  space-separated `str` (`"speed of light"`), not a list of tokens.
- **First-call downloads.** The initial use of a language fetches a Brill tagger
  and an `nltk` tokenizer over the network, then caches them. Subsequent calls
  are offline.
- **Trainer reads from the CWD.** `train.py` opens `keywords_<lang>.txt` /
  `sentences_<lang>.txt` relative to the current directory — run it from inside
  `train/`.

## Where next

- [quickstart.md](quickstart.md) — install and first call
- [api.md](api.md) — signatures and return shapes
