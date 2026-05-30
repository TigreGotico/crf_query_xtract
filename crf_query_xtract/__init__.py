from typing import List
import os
import joblib

from crf_query_xtract.features import tokenize, sent2features


class SearchtermExtractorCRF:
    """A CRF that labels the search-term tokens in a query.

    Given a natural-language query it returns the salient search keywords, e.g.
    ``"what is the speed of light"`` -> ``"the speed of light"``. Tokenisation and
    features are POS-free (see :mod:`crf_query_xtract.features`); the only runtime
    dependencies are ``joblib`` and ``sklearn_crfsuite``.
    """

    def __init__(self, lang: str = None):
        self.lang = lang
        self.model = None

    @staticmethod
    def from_pretrained(lang: str) -> "SearchtermExtractorCRF":
        lang = lang.split("-")[0].lower()
        xtractor = SearchtermExtractorCRF(lang)
        xtractor.load(f"{os.path.dirname(__file__)}/kx_{lang}.pkl")
        return xtractor

    def extract_keyword(self, text: str) -> str:
        """Return the extracted search term (``""`` when there is none)."""
        tokens = tokenize(text)
        if not tokens:
            return ""
        labels = self.model.predict([sent2features(tokens)])[0]

        keywords, current = [], []
        for word, label in zip(tokens, labels):
            if label != "O":
                current.append(word)
            elif current:
                keywords.append(" ".join(current))
                current = []
        if current:
            keywords.append(" ".join(current))
        return " ".join(k for k in keywords if k)

    def load(self, path):
        self.model = joblib.load(path)


if __name__ == "__main__":
    kx = SearchtermExtractorCRF.from_pretrained("en")
    for sentence in ["who invented the telephone",
                     "what is the speed of light",
                     "who discovered fire"]:
        print(sentence, "->", repr(kx.extract_keyword(sentence)))
