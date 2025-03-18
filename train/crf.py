import random
from typing import List, Tuple
import joblib
from sklearn_crfsuite import CRF
from ovos_utils.bracket_expansion import expand_template
from ovos_utils.list_utils import flatten_list
from crf_query_xtract import SearchtermExtractorCRF


class Trainer(SearchtermExtractorCRF):

    def _load_data(self) -> None:
        """
        Loads and preprocesses keywords and sentence templates from files.
        Files should be named keywords_<lang>.txt and sentences_<lang>.txt.
        """
        try:
            # Read keywords and expand templates
            with open(f"keywords_{self.lang}.txt", "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                self._keywords = flatten_list(expand_template(line) for line in set(lines))

            # Read sentence templates and expand them
            with open(f"sentences_{self.lang}.txt", "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                self._dataset = flatten_list(expand_template(line) for line in set(lines))

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e.filename}")

    def _generate_tagged_sentences(self, num: int = 0) -> List[List[Tuple[str, str, str]]]:
        """
        Generates a list of tagged sentences.
        Each sentence is tokenized and each token is tagged as 'K' if it is part of a keyword or 'O' otherwise.

        Args:
            num: Maximum number of sentences to generate; if negative, generate all.

        Returns:
            A list of sentences with tokens as tuples (word, POS, label).
        """
        tagged_sentences = []
        dataset_copy = self._dataset.copy()
        random.shuffle(dataset_copy)
        if num == 0:
            num = len(dataset_copy)
        for idx, template in enumerate(dataset_copy):
            keyword = random.choice(self._keywords)
            sentence = template.replace("{keyword}", keyword)
            pos_tags = self.tagger.tag(sentence)

            keyword_tokens = set(keyword.split())
            token_tags = [
                (word, pos, 'K' if word in keyword_tokens else 'O')
                for word, pos in pos_tags
            ]
            tagged_sentences.append(token_tags)

            if num > 0 and idx + 1 >= num:
                return tagged_sentences

        return tagged_sentences

    def _sent2labels(self, sent: List[Tuple[str, str, str]]) -> List[str]:
        """
        Extracts the label (K or O) for each token in the sentence.

        Args:
            sent: List of tuples (word, POS, label).

        Returns:
            A list of labels for the sentence.
        """
        return [label for _, _, label in sent]

    def train(self) -> None:
        """
        Trains the CRF model using the generated tagged sentences.
        """
        # Load data if not already loaded
        if not self._dataset or not self._keywords:
            self._load_data()

        tagged_sentences = self._generate_tagged_sentences(num=-1) + self._generate_tagged_sentences(num=-1)

        X_train = [self._sent2features([(word, pos) for word, pos, _ in sent]) for sent in tagged_sentences]
        y_train = [self._sent2labels(sent) for sent in tagged_sentences]

        self.model = CRF(algorithm='lbfgs', max_iterations=3000)
        self.model.fit(X_train, y_train)

    def save(self, path):
        joblib.dump(self.model, path)


if __name__ == "__main__":
    for lang in ["da","de","fr","it","gl", "en", "ca", "eu", "pt"]:

        kx = Trainer(lang)
        kx.train()
        kx.save(f"kx_{lang}.pkl")
        #kx.load(f"kx_{lang}.pkl")


        #kx = SearchtermExtractorCRF.from_pretrained(lang)

        # Generate a few test sentences
        #kx._load_data()
        test_sentences = kx._generate_tagged_sentences(20)

        for tagged_sentence in test_sentences:
            sentence_text = " ".join(word for word, _, _ in tagged_sentence)
            extracted = kx.extract_keyword(sentence_text)
            print("Sentence:", sentence_text)
            print("Extracted keywords:", extracted)
            print("------")
