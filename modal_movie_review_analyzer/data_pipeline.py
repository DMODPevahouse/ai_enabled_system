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


class ETL_Pipeline:
    """
    ETL_Pipeline class for extracting, transforming, and loading data.
    """

    def __init__(self):
        """
        Initializes the ETL_Pipeline class.
        """
        self.data = None

    def extract(self, filename):
        """
        Extracts data from a .csv file.
        
        Parameters:
        filename (str): The filename of the .csv file.
        
        Returns:
        None
        """
        self.data = pd.read_csv(filename)

    def transform(self):
        """
        Transforms the extracted data by cleaning, processing, and preparing it for modeling.
        
        Returns:
        None
        """

        to_export_df = self.data.copy()

        to_export_df.drop(columns='author', inplace=True)
        to_export_df.drop(columns='bought_together', inplace=True)
        to_export_df.drop(columns='store', inplace=True)
        to_export_df.drop(columns='videos', inplace=True)
        to_export_df.drop(columns='images_x', inplace=True)
        to_export_df.drop(columns='price', inplace=True)
        to_export_df.drop(columns='description', inplace=True)
        to_export_df.drop(columns='features', inplace=True)
        to_export_df.drop(columns='timestamp', inplace=True)
        to_export_df.drop(columns='user_id', inplace=True)
        to_export_df.drop(columns='asin', inplace=True)
        to_export_df.drop(columns='parent_asin', inplace=True)
        to_export_df.drop(columns='Unnamed: 0', inplace=True)
        to_export_df.drop(columns='movie_title', inplace=True)
        to_export_df.drop(columns='subtitle', inplace=True)
        to_export_df.drop(columns='images_y', inplace=True)
        to_export_df.drop(columns='details', inplace=True)
        to_export_df.drop(columns='categories', inplace=True)
        to_export_df = to_export_df.set_index('verified_purchase')

        # check if there are any rows with False values in the "verified_purchase" column
        if False in to_export_df.index:
            to_export_df = to_export_df.drop(False)

        to_export_df = to_export_df.reset_index(drop=True)


        self.data = to_export_df
        

    def load(self, output_filename):
        """
        Saves the transformed data to a .csv file.
        
        Parameters:
        output_filename (str): The filename of the output .csv file.
        
        Returns:
        None
        """
        self.data.to_csv(output_filename, index=False)
        
        



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