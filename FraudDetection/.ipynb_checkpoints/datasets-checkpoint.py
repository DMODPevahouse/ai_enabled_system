from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class Fraud_Dataset:
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data
        self.split_data = None

    def get_training_dataset(self):
        """
        Returns the training dataset for the given fold.
        """
        return self.split_data[0], self.split_data[3]

    def get_testing_dataset(self):
        """
        Returns the testing dataset for the given fold.
        """
        return self.split_data[1], self.split_data[4]

    def get_validation_dataset(self):
        """
        Returns the validation dataset for the given fold.
        """
        return self.split_data[2], self.split_data[5]

    def stratified_split(self, random_state=42, train_proportion=0.8, test_proportion=0.1, validate_proportion=0.1):
        """
        Splits a dataset into stratified train, test, and validate sets.

        Parameters:
        data (pandas DataFrame): The input data.
        labels (pandas Series): The labels corresponding to the input data.
        train_proportion (float): The proportion of data to allocate to the training set.
        test_proportion (float): The proportion of data to allocate to the test set.
        validate_proportion (float): The proportion of data to allocate to the validation set.

        Returns:
        train_data (pandas DataFrame): The training data.
        test_data (pandas DataFrame): The test data.
        validate_data (pandas DataFrame): The validation data.
        train_labels (pandas Series): The training labels.
        test_labels (pandas Series): The test labels.
        validate_labels (pandas Series): The validation labels.
        """

        # Split data into train and remaining sets
        train_data, remainder_data, train_labels, remainder_labels = train_test_split(self.data, self.labels, test_size=1 - train_proportion, random_state=42, stratify=self.labels)

        # Split remaining data into test and validate sets
        test_data, validate_data, test_labels, validate_labels = train_test_split(remainder_data, remainder_labels, test_size=test_proportion / (test_proportion + validate_proportion), random_state=42, stratify=remainder_labels)

        self.split_data = [train_data, test_data, validate_data, train_labels, test_labels, validate_labels]
