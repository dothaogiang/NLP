from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.core.dataset_loaders import load_raw_text_data

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

# Sample corpus
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]
matrix = vectorizer.fit_transform(corpus)
print("Learned Vocabulary:", vectorizer.vocabulary_)
print("Document-Term Matrix:", matrix)

# UD sample
dataset_path = "data/UD_English-EWT/en_ewt-ud-train.txt"
raw_text = load_raw_text_data(dataset_path)[:1000]
sample_corpus = [sent.strip() for sent in raw_text.split('.') if sent.strip()][:5]
ud_matrix = vectorizer.fit_transform(sample_corpus)
print("\nUD Sample Vocabulary:", vectorizer.vocabulary_)
print("UD Sample Matrix:", ud_matrix)