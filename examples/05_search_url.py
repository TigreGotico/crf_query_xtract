"""Example — turn a question into a search URL, with an empty-result guard.

Run::

    python examples/05_search_url.py
"""
from urllib.parse import quote

from crf_query_xtract import SearchtermExtractorCRF


def search_url(kx: SearchtermExtractorCRF, text: str) -> str:
    term = kx.extract_keyword(text)
    if not term:                 # no keyword and no noun fallback
        term = text              # query the raw utterance instead
    return f"https://en.wikipedia.org/w/index.php?search={quote(term)}"


def main() -> None:
    kx = SearchtermExtractorCRF.from_pretrained("en")
    for q in ["what is the speed of light", "who invented the telephone"]:
        print(q)
        print("  ->", search_url(kx, q))


if __name__ == "__main__":
    main()
