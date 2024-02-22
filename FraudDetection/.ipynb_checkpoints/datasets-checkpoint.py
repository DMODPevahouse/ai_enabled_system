import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import List, Tuple


class Fraud_Dataset:
    def __init__(self, data: pd.DataFrame, labels: pd.Series, splits=5, shuffled: bool=False, random: int=None):
        """
        Initializes the Fraud_Dataset class with the input data and labels.

        Parameters:
        data (pandas DataFrame): The input data.
        labels (pandas Series): The labels corresponding to the input data.
        """
        self.labels = labels
        self.data = data
        self.kf = StratifiedKFold(n_splits=splits, shuffle=shuffled, random_state=random)  # Initialize k-fold cross validation with 5 folds
        self.split_data = None

    def stratified_split(self, random_state: int = 42, train_proportion: float = 0.8, test_proportion: float = 0.1, validate_proportion: float = 0.1) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """
        Splits a dataset into stratified train, test, and validate sets. This is configured to also be split into folds.

        Parameters:
        data (pandas DataFrame): The input data.
        labels (pandas Series): The labels corresponding to the input data.
        train_proportion (float): The proportion of data to allocate to the training set.
        test_proportion (float): The proportion of data to allocate to the test set.
        validate_proportion (float): The proportion of data to allocate to the validation set.

        Returns:
        List[Tuple[pandas DataFrame, pandas Series]]: A list of tuples containing the training, test, and validation datasets and their corresponding labels.
        """
        split_data = []
        for train_index, test_index in self.kf.split(self.data, self.labels):
            validate_data, test_data, validate_labels, test_labels = train_test_split(self.data.iloc[test_index], self.labels.iloc[test_index], test_size=.5, random_state=random_state, stratify=self.labels.iloc[test_index])
            split_data.append((self.data.iloc[train_index], validate_data, test_data, self.labels.iloc[train_index], validate_labels, test_labels))
        self.split_data = split_data

    def get_training_dataset(self, fold: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns the training dataset for the given fold.

        Parameters:
        fold (int): The fold number.

        Returns:
        Tuple[pandas DataFrame, pandas Series]: The training dataset and its corresponding labels.
        """
        return self.split_data[fold][0], self.split_data[fold][3]

    def get_testing_dataset(self, fold: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns the testing dataset for the given fold.

        Parameters:
        fold (int): The fold number.

        Returns:
        Tuple[pandas DataFrame, pandas Series]: The testing dataset and its corresponding labels.
        """
        return self.split_data[fold][2], self.split_data[fold][5]

    def get_validation_dataset(self, fold: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns the validation dataset for the given fold.

        Parameters:
        fold (int): The fold number.

        Returns:
        Tuple[pandas DataFrame, pandas Series]: The validation dataset and its corresponding labels.
        """
        return self.split_data[fold][1], self.split_data[fold][4]
