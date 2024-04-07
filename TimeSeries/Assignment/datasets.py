import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple

#unused, ended up being difficult to implement
class TimeSeriesFraudDataset:
    def __init__(self, data: pd.DataFrame, splits=5, shuffled: bool=False, random: int=None):
        """
        Initializes the TimeSeriesFraudDataset class with the input data and labels.

        :param data (pandas DataFrame): The input data.
        :param splits: optional variable to set how many splits can happen for testing
        :param shuffled: gives the option to shuffle the splits
        :param random
        """
        self.data = data
        self.tscv = TimeSeriesSplit(n_splits=splits)  # Initialize Time Series Cross Validation with 5 folds
        self.split_data = None

    def time_series_split(self) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """
        Splits a time series dataset into train, test, and validate sets using Time Series Cross Validation.

        :param random_state: Capability of choosing the random state of the stratified split

        Returns: No return but places this data into the model:
        List[Tuple[pandas DataFrame, pandas Series]]: A list of tuples containing the training, test, and validation datasets and their corresponding labels.
        """
        split_data = []
        for train_index, test_index in self.tscv.split(self.data):
            train_data, test_data = self.data.iloc[train_index], self.data.iloc[test_index]
            split_data.append((train_data, test_data))
        self.split_data = split_data

    def get_training_dataset(self, fold: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns the training dataset for the given fold.

        Parameters:
        fold (int): The fold number.

        Returns:
        Tuple[pandas DataFrame, pandas Series]: The training dataset and its corresponding labels.
        """
        return self.split_data[fold][0]

    def get_testing_dataset(self, fold: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns the testing dataset for the given fold.

        Parameters:
        fold (int): The fold number.

        Returns:
        Tuple[pandas DataFrame, pandas Series]: The testing dataset and its corresponding labels.
        """
        return self.split_data[fold][1]

    def get_validation_dataset(self, fold: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns the validation dataset for the given fold.

        Parameters:
        fold (int): The fold number.

        Returns:
        Tuple[pandas DataFrame, pandas Series]: The validation dataset and its corresponding labels.
        """
        raise NotImplementedError("Time Series Cross Validation does not support validation sets.")