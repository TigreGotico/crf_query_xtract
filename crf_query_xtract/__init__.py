from typing import List, Tuple
import os
import joblib
from brill_postaggers import BrillPostagger


class SearchtermExtractorCRF:
    """
    A CRF-based keyword extractor that uses keywords and sentence templates to train a model.
    """
    def __init__(self, lang: str):
        """
        Initialize the extractor with a given language.
        Loads the keywords and sentences datasets and trains the CRF model.
        """
        self.tagger = BrillPostagger.from_pretrained(lang)
        self.lang: str = lang
        self._keywords: List[str] = []
        self._dataset: List[str] = []
        self.model = None

    @staticmethod
    def from_pretrained(lang: str):
        lang = lang.split("-")[0].lower()
        model = f"kx_{lang}"
        xtractor = SearchtermExtractorCRF(lang)
        xtractor.load(f"{os.path.dirname(__file__)}/{model}.pkl")
        return xtractor

    def _word2features(self, sent: List[Tuple[str, str]], idx: int) -> dict:
        """
        Extracts features for a given word in a sentence.

        Args:
            sent: List of tuples (word, POS) for the sentence.
            idx: Index of the token in the sentence.

        Returns:
            A feature dictionary for the token at the given index.
        """
        word, pos = sent[idx]
        features = {
            'word.lower()': word.lower(),
            'pos': pos,
        }

        # Previous features
        if idx > 0:
            prev_word, prev_pos = sent[idx - 1]
            features.update({
                '-1:word.lower()': prev_word.lower(),
                '-1:pos': prev_pos,
            })
        else:
            features['BOS'] = True  # Beginning of Sentence

        # Two words back
        if idx > 1:
            prev2_word, prev2_pos = sent[idx - 2]
            features.update({
                '-2:word.lower()': prev2_word.lower(),
                '-2:pos': prev2_pos,
            })

        # Next features
        if idx < len(sent) - 1:
            next_word, next_pos = sent[idx + 1]
            features.update({
                '+1:word.lower()': next_word.lower(),
                '+1:pos': next_pos,
            })
        else:
            features['EOS'] = True  # End of Sentence

        if idx < len(sent) - 2:
            next2_word, next2_pos = sent[idx + 2]
            features.update({
                '+2:word.lower()': next2_word.lower(),
                '+2:pos': next2_pos,
            })

        return features

    def _sent2features(self, sent: List[Tuple[str, str]]) -> List[dict]:
        """
        Converts a sentence into a list of feature dictionaries.

        Args:
            sent: List of tuples (word, POS).

        Returns:
            A list of features for each token in the sentence.
        """
        return [self._word2features(sent, i) for i in range(len(sent))]

    def extract_keyword(self, text: str) -> str:
        """
        Extracts keywords from a given text using the trained CRF model.

        Args:
            text: The input text.

        Returns:
            A string containing the extracted keywords.
        """
        pos_tags = self.tagger.tag(text)
        features = self._sent2features(pos_tags)
        predicted_labels = self.model.predict([features])[0]

        extracted_keywords = []
        current_keyword = []

        for (word, _), label in zip(pos_tags, predicted_labels):
            if label == 'K':
                current_keyword.append(word)
            elif current_keyword:
                extracted_keywords.append(" ".join(current_keyword))
                current_keyword = []
        if current_keyword:
            extracted_keywords.append(" ".join(current_keyword))
        extracted_keywords = [k for k in extracted_keywords if k]

        # return first noun as fallback
        if not extracted_keywords:
            for word, tag in pos_tags:
                if tag == 'NOUN':
                    return word

        return " ".join(extracted_keywords)

    def load(self, path):
        self.model = joblib.load(path)


if __name__ == "__main__":

    kx = SearchtermExtractorCRF.from_pretrained("en")

    test_sentences = ["who invented the telephone",
                      "what is the speed of light",
                      "who discovered fire"]

    for sentence_text  in test_sentences:
        extracted = kx.extract_keyword(sentence_text)
        print("Sentence:", sentence_text)
        print("Extracted keywords:", extracted)
        print("------")
