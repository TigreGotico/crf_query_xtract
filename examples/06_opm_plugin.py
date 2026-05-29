"""Example — the OVOS plugin shape: {keyword: confidence}.

Run::

    python examples/06_opm_plugin.py
"""
from typing import Dict

from crf_query_xtract import SearchtermExtractorCRF


def extract(kx: SearchtermExtractorCRF, text: str) -> Dict[str, float]:
    """Mirror the plugin's {keyword: 1.0} output from a loaded model."""
    keyword = kx.extract_keyword(text)
    return {keyword: 1.0} if keyword else {}


def main() -> None:
    # The plugin advertises which languages it serves. Importing it pulls in
    # the OVOS plugin-manager stack; skip cleanly if that is not installed.
    try:
        from crf_query_xtract.opm import CRFBrillKeywordExtractor
        print("supported langs:", sorted(CRFBrillKeywordExtractor().supported_langs))
    except ImportError as exc:
        print(f"OPM stack unavailable, skipping plugin metadata ({exc})")

    # Produce the same mapping the plugin returns, from a loaded model.
    kx = SearchtermExtractorCRF.from_pretrained("en")
    for q in ["who invented the telephone", "what is the speed of light"]:
        print(f"{q!r:38} -> {extract(kx, q)}")


if __name__ == "__main__":
    main()
