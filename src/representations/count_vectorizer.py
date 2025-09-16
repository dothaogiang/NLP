from collections import defaultdict
from src.core.interfaces import Vectorizer
from src.preprocessing.regex_tokenizer import RegexTokenizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or RegexTokenizer()
        self.vocabulary_ = {}

    def fit(self, corpus: list[str]):
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)
        sorted_tokens = sorted(unique_tokens)
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted_tokens)}

    def transform(self, documents: list[str]) -> list[list[int]]:
        if not self.vocabulary_:
            raise ValueError("Vectorizer not fitted.")
        vocab_size = len(self.vocabulary_)
        matrix = []
        for doc in documents:
            counts = defaultdict(int)
            for token in self.tokenizer.tokenize(doc):
                if token in self.vocabulary_:
                    counts[self.vocabulary_[token]] += 1
            vector = [counts[i] for i in range(vocab_size)]
            matrix.append(vector)
        return matrix