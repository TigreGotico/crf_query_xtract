"""Example — extract a search term from a full question.

Run::

    python examples/01_quickstart.py
"""
from crf_query_xtract import SearchtermExtractorCRF


def main() -> None:
    kx = SearchtermExtractorCRF.from_pretrained("en")

    questions = [
        "who invented the telephone",
        "what is the speed of light",
        "who discovered fire",
    ]
    for q in questions:
        print(f"{q!r:42} -> {kx.extract_keyword(q)!r}")


if __name__ == "__main__":
    main()
