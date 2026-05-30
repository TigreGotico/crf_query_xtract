"""Example — locale codes resolve to a base model, and the noun fallback.

Run::

    python examples/04_locale_and_fallback.py
"""
from crf_query_xtract import SearchtermExtractorCRF


def main() -> None:
    # A locale code is normalized: region is dropped, case lowered.
    # "pt-BR" loads the same model as "pt".
    kx = SearchtermExtractorCRF.from_pretrained("pt-BR")
    print("resolved lang:", kx.lang)
    print("pt-BR ->", repr(kx.extract_keyword("quem inventou o telefone")))

    # When the model labels nothing as a keyword, extract_keyword falls back
    # to the first noun in the sentence (and returns "" if there is none).
    en = SearchtermExtractorCRF.from_pretrained("en")
    for text in ["fire", "telephone", "speed of light"]:
        print(f"{text!r:18} -> {en.extract_keyword(text)!r}")


if __name__ == "__main__":
    main()
