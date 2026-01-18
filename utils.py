import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """
    Cleans the text by removing URLs and special characters,
    and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove special chars (keep only alphanumeric and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.lower().strip()

def tokenizer_porter(text: str) -> list:
    """
    Splits the text into words and applies Porter Stemmer (root form).
    """
    return [stemmer.stem(word) for word in text.split()]