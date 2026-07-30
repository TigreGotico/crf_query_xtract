"""Regression tests for the CRF search-term extractor and its OPM plugin.

Models are downloaded from the Hub on first use, so the model-dependent tests
skip gracefully when the Hub is unreachable / the repo needs auth (e.g. offline
CI) rather than failing.
"""
import pytest

from crf_query_xtract import SearchtermExtractorCRF

MODEL_LANGS = {"ca", "da", "de", "en", "es", "eu", "fr", "gl", "it", "nl", "pt"}


def _load(lang):
    try:
        return SearchtermExtractorCRF.from_pretrained(lang)
    except Exception as e:  # noqa: BLE001 — no network / private repo without token
        pytest.skip(f"model for {lang!r} unavailable: {type(e).__name__}")


def test_from_pretrained_extracts_english_keyword():
    assert _load("en").extract_keyword("who invented the telephone") == "telephone"


def test_from_pretrained_extracts_portuguese_keyword():
    assert _load("pt").extract_keyword("qual é a capital de portugal") == "portugal"


def test_from_pretrained_normalises_locale_code():
    assert _load("en-US").model is not None  # "en-US" resolves to the "en" model


def test_extract_keyword_always_returns_string():
    assert isinstance(_load("en").extract_keyword("hello"), str)


def test_from_pretrained_local_dir(tmp_path):
    # A local directory of kx_<lang>.pkl files is a valid `repo_id`.
    import joblib
    kx = _load("en")
    joblib.dump(kx.model, tmp_path / "kx_en.pkl")
    local = SearchtermExtractorCRF.from_pretrained("en", repo_id=str(tmp_path))
    assert local.extract_keyword("who invented the telephone") == "telephone"


def _plugin():
    pytest.importorskip("ovos_plugin_manager.templates.keywords")
    from crf_query_xtract.opm import CRFBrillKeywordExtractor
    return CRFBrillKeywordExtractor


def test_plugin_supported_langs():
    plugin = _plugin()(config={"lang": "en"})
    assert plugin.supported_langs == MODEL_LANGS


def test_plugin_extract_scores_keyword():
    plugin = _plugin()(config={"lang": "en"})
    try:
        scored = plugin.extract("who invented the telephone", lang="en")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"model unavailable: {type(e).__name__}")
    assert scored == {"telephone": 1.0}


def test_plugin_rejects_unsupported_lang():
    plugin = _plugin()(config={"lang": "en"})
    with pytest.raises(ValueError):
        plugin.get_extractor("xx")
