import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data_pipeline import ETL_Pipeline
from datasets import Modal_Dataset
from metrics import Metrics
from sklearn.preprocessing import MinMaxScaler
import os

class Movie_Review_Model:
    """
    Fraud Detector Model Class:
    This class is responsible for constructing the model, handling the necessary logic to take raw input data,
    and produce an output.
    """

    def __init__(self, data_file, model=LogisticRegression(max_iter=10000), n_splits=5):
        """
        Initialize the Fraud Detector Model.
        :param model: The model to be used for fraud detection. Default is RandomForestClassifier.
        :param n_splits: The number of folds to split the dataset into for cross-validation. Default is 5.
        """
        self.model = model
        self.n_splits = n_splits
        self.data_file = data_file
        self.data = None
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', self.model)
        ])
        self.etl = ETL_Pipeline()
        
    def initiate_etl(self):
        """
        Initialize and run the ETL pipeline, loading the transformed data into a DataFrame.
        """
        self.etl.extract(self.data_file)
        self.etl.transform()
        if not os.path.exists('testing.csv'):
            os.makedirs('encoded_review.csv')
            self.etl.load("encoded_review.csv")
            self.data = pd.read_csv("encoded_review.csv")
        X = self.etl.data.drop('rating', axis=1)
        y = self.etl.data['rating']
        return X, y

    def data_fold(self, X, y):
        """
        Create a Fraud_Dataset object, perform stratified splitting, and return the folded dataset.
        :param X: The feature set or independent variables.
        :param y: The target variable or dependent variable.
        """
        folded_data = Modal_Dataset(X, y)
        folded_data.stratified_split()
        return folded_data

    def train(self, cross_validate=False, new_file=False):
        """
        Train the model using the provided training data.
        :param X: The feature set or independent variables.
        :param y: The target variable or dependent variable.
        """
        if new_file:
            X_nonfold, y_nonfold = self.initiate_etl()
        else:
            self.data = pd.read_csv('testing.csv')
            X_nonfold = self.etl.data.drop('rating', axis=1)
            y_nonfold = self.etl.data['rating']
        if cross_validate:
            split_set = self.data_fold(X_nonfold, y_nonfold)
            self.cross_validation(split_set)
        else:
            self.pipeline.fit(X_nonfold, y_nonfold)
    
    def cross_validation(self, split_data):
        """
        Perform k-fold cross-validation on the dataset using the given metric function.
        :param split_data: data that is split up for kfold validation
        :param metric_func: the function that will be used to determine what metrics are going to be used
        :return: The classification report
        """
        metric_func = Metrics()
        for i in range(self.n_splits):
            X, y = split_data.get_training_dataset(i)
            X_test, y_test = split_data.get_testing_dataset(i)
            X_validation, y_validation = split_data.get_validation_dataset(i)
            self.pipeline.fit(X, y)
            self.test(X, y, i, "train", metric_func)
            self.test(X_test, y_test, i, "test", metric_func)
            self.test(X_validation, y_validation, i, "validation", metric_func)
        
    def test(self, X, y, fold, iteration, metric_func):
        """
        Test the model using the provided testing data and calculate the metrics.
        :param X: The feature set or independent variables.
        :param y: The target variable or dependent variable.
        """
        y_pred = self.pipeline.predict_proba(X)
        metric_func.run(y_pred, y, fold, iteration)

        
    def json_predict(self, info):
        """
        Takes the data to be predicted on and produces a prediction. This specific function is different from the predict due to being a different format
        :param info: data to be used to get a predicion
        """
        info.to_csv("predictions.csv", index=False)
        self.etl.extract("predictions.csv")
        self.etl.transform(prediction=True)
        self.etl.load("transformed_predictions.csv", create_file=True)
        info = pd.read_csv("transformed_predictions.csv")
        info.drop(columns=['rating'], inplace=True)
        #info = info.reset_index(drop=True)
        prediction = self.pipeline.predict(info)
        return prediction


