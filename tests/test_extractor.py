"""Regression tests for the CRF search-term extractor and its OPM plugin."""
import glob
import os

import pytest

from crf_query_xtract import SearchtermExtractorCRF

HERE = os.path.dirname(os.path.dirname(__file__))
SHIPPED_LANGS = {
    os.path.basename(p)[len("kx_"):-len(".pkl")]
    for p in glob.glob(os.path.join(HERE, "crf_query_xtract", "kx_*.pkl"))
}


def test_models_shipped_for_expected_langs():
    assert SHIPPED_LANGS == {"ca", "da", "de", "en", "eu", "fr", "gl", "it", "pt"}


def test_from_pretrained_extracts_english_keyword():
    kx = SearchtermExtractorCRF.from_pretrained("en")
    assert kx.extract_keyword("who invented the telephone") == "telephone"


def test_from_pretrained_extracts_portuguese_keyword():
    kx = SearchtermExtractorCRF.from_pretrained("pt")
    assert kx.extract_keyword("qual é a capital de portugal") == "portugal"


def test_from_pretrained_normalises_locale_code():
    # "en-US" must resolve to the "en" model, not raise.
    kx = SearchtermExtractorCRF.from_pretrained("en-US")
    assert kx.model is not None


def test_extract_keyword_always_returns_string():
    kx = SearchtermExtractorCRF.from_pretrained("en")
    out = kx.extract_keyword("hello")
    assert isinstance(out, str)


def _plugin():
    # The OPM template pulls in ovos_plugin_manager; skip if it can't import here.
    pytest.importorskip("ovos_plugin_manager.templates.keywords")
    from crf_query_xtract.opm import CRFBrillKeywordExtractor
    return CRFBrillKeywordExtractor


def test_plugin_supported_langs_match_shipped_models():
    plugin = _plugin()(config={"lang": "en"})
    assert plugin.supported_langs == SHIPPED_LANGS


def test_plugin_extract_scores_keyword():
    # Exercises the from_pretrained loading path end to end.
    plugin = _plugin()(config={"lang": "en"})
    scored = plugin.extract("who invented the telephone", lang="en")
    assert scored == {"telephone": 1.0}


def test_plugin_rejects_unsupported_lang():
    plugin = _plugin()(config={"lang": "en"})
    with pytest.raises(ValueError):
        plugin.get_extractor("xx")
