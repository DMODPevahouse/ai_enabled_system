import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def preprocess(corpus):
    """
    Preprocesses a pandas.Series() of text data into an indexed vocabulary and preprocessed tokens.

    Args:
    corpus (pandas.Series()): A Series of text data.

    Returns:
    vocab (dict): A dictionary of unique words in the corpus.
    tokens (list): A list of preprocessed tokens in the corpus.
    """
    # Initialize empty lists to store vocabulary and tokens
    vocab = {}
    tokens = []

    # Tokenize the text data and update the vocabulary
    for text in corpus:
        tokens_text = word_tokenize(text.lower())
        tokens.extend(tokens_text)
        for word in tokens_text:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab, tokens

def encode(corpus, encoding_method):
    """
    Encodes a pandas.Series() of text data using a specified encoding method.

    Args:
    corpus (pandas.Series()): A Series of text data.
    encoding_method (str): The encoding method to use. Can be 'bow', 'tfidf', or 'word2vec'.

    Returns:
    encoded_data (numpy.ndarray): The encoded data using the specified method.
    """
    # Preprocess the corpus
    vocab, tokens = preprocess(corpus)

    # Initialize the encoded data
    encoded_data = None

    # Encode the corpus using the specified method
    if encoding_method == 'bow':
        # Bag-of-Words encoding
        bow_vectorizer = CountVectorizer(vocabulary=vocab)
        bow_matrix = bow_vectorizer.fit_transform(corpus)
        encoded_data = bow_matrix.toarray()

    elif encoding_method == 'tfidf':
        # TF-IDF encoding
        tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        encoded_data = tfidf_matrix.toarray()

    elif encoding_method == 'word2vec':
        # Word2Vec encoding
        word2vec_model = Word2Vec([word_tokenize(text.lower()) for text in corpus], size=100, window=5, min_count=1, workers=4)
        encoded_data = []
        for text in corpus:
            vector = np.zeros(100)
            words = word_tokenize(text.lower())
            for word in words:
                if word in word2vec_model.wv:
                    vector += word2vec_model.wv[word]
            if len(words) > 0:
                vector /= len(words)
            encoded_data.append(vector)
        encoded_data = np.array(encoded_data)

    return encoded_data