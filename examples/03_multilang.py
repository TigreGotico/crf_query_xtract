"""Example — one question phrased across several supported languages.

Run::

    python examples/03_multilang.py
"""
from crf_query_xtract import SearchtermExtractorCRF

# "who invented the telephone", per language.
QUESTIONS = {
    "en": "who invented the telephone",
    "fr": "qui a invente le telephone",
    "it": "chi ha inventato il telefono",
    "gl": "quen inventou o telefono",
    "ca": "qui va inventar el telefon",
    "pt": "quem inventou o telefone",
}


def main() -> None:
    for lang, question in QUESTIONS.items():
        kx = SearchtermExtractorCRF.from_pretrained(lang)
        keyword = kx.extract_keyword(question)
        print(f"[{lang}] {question!r:38} -> {keyword!r}")


if __name__ == "__main__":
    main()
