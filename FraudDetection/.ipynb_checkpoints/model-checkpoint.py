import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data_pipeline import ETL_Pipeline
from datasets import Fraud_Dataset
from metrics import Metrics
from sklearn.preprocessing import MinMaxScaler

class Fraud_Detector_Model:
    """
    Fraud Detector Model Class:
    This class is responsible for constructing the model, handling the necessary logic to take raw input data,
    and produce an output.
    """

    def __init__(self, data_file, model=RandomForestClassifier(), n_splits=5):
        """
        Initialize the Fraud Detector Model.
        :param model: The model to be used for fraud detection. Default is RandomForestClassifier.
        :param n_splits: The number of folds to split the dataset into for cross-validation. Default is 5.
        """
        self.model = model
        self.n_splits = n_splits
        self.data = data_file
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', self.model)
        ])
        self.scaler = None
        
    def initiate_etl(self):
        """
        Initialize and run the ETL pipeline, loading the transformed data into a DataFrame.
        """
        etl = ETL_Pipeline()
        etl.extract(self.data)
        etl.transform()
        etl.load("transformed_transactions.csv")
        self.data = pd.read_csv("transformed_transactions.csv")
        self.initiate_scaler()
        X = self.data.drop('is_fraud', axis=1)
        y = self.data['is_fraud']
        return X, y

    def data_fold(self, X, y):
        """
        Create a Fraud_Dataset object, perform stratified splitting, and return the folded dataset.
        :param X: The feature set or independent variables.
        :param y: The target variable or dependent variable.
        """
        folded_data = Fraud_Dataset(X, y)
        folded_data.stratified_split()
        return folded_data

    def train(self, cross_validate=False):
        """
        Train the model using the provided training data.
        :param X: The feature set or independent variables.
        :param y: The target variable or dependent variable.
        """
        X_nonfold, y_nonfold = self.initiate_etl()
        split_set = self.data_fold(X_nonfold, y_nonfold)
        metric_time = Metrics()
        if cross_validate:
            self.cross_validation(split_set, metric_time)
        else:
            self.pipeline.fit(X_nonfold, y_nonfold)
    
    def cross_validation(self, split_data, metric_func):
        """
        Perform k-fold cross-validation on the dataset using the given metric function.
        :param split_data: data that is split up for kfold validation
        :param metric_func: the function that will be used to determine what metrics are going to be used
        """
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
        :return: The classification report and confusion matrix.
        """
        y_pred = self.pipeline.predict(X)
        metric_func.run(y, y_pred, fold, iteration)

    def predict(self, info):
        info = self.scaler.transform(info)
        prediction = self.pipeline.predict(info)
        return prediction


    def initiate_scaler(self):
        self.scaler = MinMaxScaler()
        self.data = pd.DataFrame(self.scaler.fit_transform(self.data), columns=self.data.columns)

