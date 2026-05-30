"""Example — keyword extraction on Portuguese questions.

Run::

    python examples/02_portuguese.py
"""
from crf_query_xtract import SearchtermExtractorCRF


def main() -> None:
    kx = SearchtermExtractorCRF.from_pretrained("pt")

    questions = [
        "quem inventou o telefone",
        "qual a velocidade da luz",
        "quem descobriu o fogo",
        "qual a capital de Portugal",
    ]
    for q in questions:
        print(f"{q!r:38} -> {kx.extract_keyword(q)!r}")


if __name__ == "__main__":
    main()
