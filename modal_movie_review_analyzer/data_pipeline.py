import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import csr_matrix
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')

class ETL_Pipeline:
    """
    ETL_Pipeline class for extracting, transforming, and loading data.
    """

    def __init__(self):
        """
        Initializes the ETL_Pipeline class.
        """
        self.data = None
        self.vectorizer = None

    def extract(self, filename):
        """
        Extracts data from a .csv file.
        
        Parameters:
        filename (str): The filename of the .csv file.
        
        Returns:
        None
        """
        self.data = pd.read_csv(filename, low_memory=False)

    def transform(self, prediction=False):
        """
        Transforms the extracted data by cleaning, processing, and preparing it for modeling.
        
        Returns:
        None
        """

        to_export_df = self.data.copy()

        columns_to_drop = ['author', 'bought_together', 'store', 'videos', 'images_x', 'price',
                   'description', 'features', 'timestamp', 'user_id', 'asin',
                   'parent_asin', 'Unnamed: 0', 'movie_title', 'subtitle',
                   'images_y', 'details', 'categories', 'review_title', 'verified_purchase']

        for col in columns_to_drop:
            if col in to_export_df.columns:
                to_export_df.drop(columns=[col], inplace=True)


        # check if there are any rows with False values in the "verified_purchase" column
        if False in to_export_df.index:
            to_export_df = to_export_df.drop(False)

        to_export_df = to_export_df.reset_index(drop=True)
        self.data = to_export_df.copy()
        self.data["text"] = self.data["text"].fillna("")
        self.preprocess_data()
        self.encode()
        if not prediction:
            self.data["rating"] = to_export_df["rating"]
        

    def load(self, output_filename, create_file=False):
        """
        Saves the transformed data to a .csv file.
        
        Parameters:
        output_filename (str): The filename of the output .csv file.
        
        Returns:
        None
        """
        if create_file:
            #self.data.to_csv(output_filename, index=False)
            chunk_size = 100000

            # Write the DataFrame to a CSV file in chunks
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                self.data.to_csv(output_filename, index=False, chunksize=chunk_size)
        
    
    def encode(self):
        if self.vectorizer == None:
            # Initialize a TfidfVectorizer object
            self.vectorizer = TfidfVectorizer(max_features=100000, min_df=20)

            # Fit and transform the text data
            self.data = self.vectorizer.fit_transform(self.data['clean_text'])
            self.data = pd.DataFrame.sparse.from_spmatrix(self.data, columns=self.vectorizer.get_feature_names_out())
        else:
            self.data = self.vectorizer.transform(self.data['clean_text'])
            self.data = pd.DataFrame.sparse.from_spmatrix(self.data, columns=self.vectorizer.get_feature_names_out())
            
    def preprocess_data(self):
        self.data['clean_text'] = preprocess(self.data['text'])
        
        

# Define a function to preprocess text data
def preprocess(data: pd.Series) -> pd.Series:
    return data.apply(custom_preprocessing)

def custom_preprocessing(text):
    # Check for empty strings or NaN values
    if pd.isna(text) or text == '':
        return ''

    # Process text using spacy
    doc = nlp(text)

    # Remove stopwords and punctuation
    tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop and not token.is_punct]

    # Join tokens back into a string
    text = ' '.join(tokens)

    return text
