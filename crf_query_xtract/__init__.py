from typing import Optional
import os
import joblib

from crf_query_xtract.features import tokenize, sent2features

#: Hub repo the per-language models are downloaded from. Override with the
#: ``CRF_QUERY_XTRACT_REPO`` env var or the ``repo_id`` argument.
DEFAULT_REPO = os.environ.get("CRF_QUERY_XTRACT_REPO", "TigreGotico/crf-query-xtract")


class SearchtermExtractorCRF:
    """A CRF that labels the search-term tokens in a query.

    Given a natural-language query it returns the salient search keywords, e.g.
    ``"what is the speed of light"`` -> ``"the speed of light"``. Tokenisation and
    features are POS-free (see :mod:`crf_query_xtract.features`).
    """

    def __init__(self, lang: str = None):
        self.lang = lang
        self.model = None

    @staticmethod
    def from_pretrained(lang: str, repo_id: Optional[str] = None) -> "SearchtermExtractorCRF":
        """Load the model for ``lang``.

        Downloads ``kx_<lang>.pkl`` from ``repo_id`` (default :data:`DEFAULT_REPO`
        on the HF Hub, cached after the first call). Pass your own ``repo_id`` —
        another Hub repo, or a local directory of ``kx_<lang>.pkl`` files — to use
        your own models. For a single file, use the bare constructor + ``load``.
        """
        lang = lang.split("-")[0].lower()
        repo_id = repo_id or DEFAULT_REPO
        fname = f"kx_{lang}.pkl"
        xtractor = SearchtermExtractorCRF(lang)
        if os.path.isdir(repo_id):
            xtractor.load(os.path.join(repo_id, fname))
        else:
            from huggingface_hub import hf_hub_download
            xtractor.load(hf_hub_download(repo_id, fname))
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
