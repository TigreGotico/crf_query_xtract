from ovos_plugin_manager.templates.keywords import KeywordExtractor
from crf_query_xtract import SearchtermExtractorCRF
from typing import Dict, Optional, Set


class CRFBrillKeywordExtractor(KeywordExtractor):

    def __init__(self, config=None):
        super().__init__(config)
        self._xtractors: Dict[str, SearchtermExtractorCRF] = {}

    @property
    def supported_langs(self) -> Set[str]:
        return {"ca", "da", "en", "eu", "fr", "gl", "it", "pt"}

    def get_extractor(self, lang: Optional[str] = None) -> SearchtermExtractorCRF:
        lang = lang or self.lang
        if lang.lower().split("-")[0] not in self.supported_langs:
            raise ValueError(f"Unsupported language: {lang}")
        if lang not in self._xtractors:
            self._xtractors[lang] = SearchtermExtractorCRF(lang)
        return self._xtractors[lang]

    def extract(self, text: str, lang: Optional[str] = None) -> Dict[str, float]:
        kx = self.get_extractor(lang)
        extracted = kx.extract_keyword(text)
        if extracted:
            return {extracted: 1.0}
        return {}
