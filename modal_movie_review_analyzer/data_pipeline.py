import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import LatentDirichletAllocation
import gensim
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

def preprocess(corpus):
    """
    This function takes in a pandas.Series() of a corpus of text data as an argument.
    This function should output an indexed vocabulary and preprocessed tokens.

    Input:
        corpus (pandas.Series): a series of text data
    Output:
        vocab (dict): a dictionary of indexed vocabulary
        tokens (list): a list of preprocessed tokens
    """
    # Initialize empty lists to store tokens and vocabulary
    vocab = {}
    tokens = []

    # Iterate over each document in the corpus
    for i, doc in enumerate(corpus):
        # Tokenize the document
        tokenized_doc = word_tokenize(doc)

        # Convert tokens to lowercase
        tokenized_doc = [token.lower() for token in tokenized_doc]

        # Filter out punctuation and non-alphanumeric characters
        tokenized_doc = [token for token in tokenized_doc if token.isalnum()]

        # Remove duplicate tokens from the document
        tokenized_doc = list(set(tokenized_doc))

        # Add document tokens to the tokens list
        tokens.append(tokenized_doc)

        # Update the vocabulary
        for token in tokenized_doc:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab, tokens

def encode(preprocessed_tokens, encoding_method):
    """
    This function takes in two arguments: 1) the preprocessed token outputs of the preprocess() function,
    and 2) the desired encoding method.
    
    It then encodes the preprocessed tokens using the specified encoding method and returns the encoded tokens.
    
    The available encoding methods are 'Bag-of-Words', 'TF-IDF', and 'Word2Vec'.
    """
    if encoding_method == 'Bag-of-Words':
        # Initialize a CountVectorizer model
        vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

        # Join the tokens of each document into a single string
        preprocessed_documents = [' '.join(doc_tokens) for doc_tokens in preprocessed_tokens]

        # Encode the preprocessed tokens using the vectorizer
        encoding = vectorizer.fit_transform(preprocessed_documents)
    elif encoding_method == 'TF-IDF':
        # Initialize a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')

        # Join the tokens of each document into a single string
        preprocessed_documents = [' '.join(doc_tokens) for doc_tokens in preprocessed_tokens]

        # Encode the preprocessed tokens using the vectorizer
        encoding = vectorizer.fit_transform(preprocessed_documents)
    elif encoding_method == 'Word2Vec':
        # Initialize a Word2Vec model
        model = gensim.models.Word2Vec(preprocessed_tokens, min_count=1)

        # Encode the preprocessed tokens using the model
        encoding = []
        for doc_tokens in preprocessed_tokens:
            doc_vec = np.zeros(model.vector_size)
            for token in doc_tokens:
                if token in model.wv:
                    doc_vec += model.wv[token]
            encoding.append(doc_vec)
        encoding = np.array(encoding)
    else:
        raise ValueError("Invalid encoding method. Choose from 'Bag-of-Words', 'TF-IDF', or 'Word2Vec'.")

    return encoding