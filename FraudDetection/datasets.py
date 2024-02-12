import numpy as np
import pandas as pd

class Fraud_Dataset:
    """
    A class representing a fraud dataset with methods for splitting and partitioning the data.
    """

    def __init__(self, data, label, test_size=0.2, validation_size=0.2, random_seed=42):
        """
        Initializes the Fraud_Dataset object with the given data and label.

        :param data: A Pandas DataFrame containing the input data.
        :param label: A Pandas Series containing the output label.
        :param test_size: The fraction of the data to use for testing. Default is 0.2.
        :param validation_size: The fraction of the data to use for validation. Default is 0.2.
        :param random_seed: The random seed to use for splitting the data. Default is 42.
        """
        self.data = data
        self.label = label
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_seed = random_seed

    def get_splitter(self):
        """
        Returns a Pandas DataFrame containing the input data and output label, split into training, validation, and test sets.

        :return: A Pandas DataFrame with columns for the input data and output label, split into training, validation, and test sets.
        """
        # Generate a random permutation of the data index
        perm_index = np.random.RandomState(seed=self.random_seed).permutation(self.data.index)

        # Split the data and labels into training, validation, and test sets
        train_data = self.data.loc[perm_index[:int((1-self.test_size-self.validation_size)*len(self.data))]]
        train_label = self.label.loc[perm_index[:int((1-self.test_size-self.validation_size)*len(self.data))]]

        val_data = self.data.loc[perm_index[int((1-self.test_size-self.validation_size)*len(self.data)):int((1-self.test_size)*len(self.data))]]
        val_label = self.label.loc[perm_index[int((1-self.test_size-self.validation_size)*len(self.data)):int((1-self.test_size)*len(self.data))]]

        test_data = self.data.loc[perm_index[int((1-self.test_size)*len(self.data)):]]
        test_label = self.label.loc[perm_index[int((1-self.test_size)*len(self.data)):]]

        # Combine the data and labels into a single DataFrame
        split_data = pd.concat([train_data, val_data, test_data], axis=0)
        split_label = pd.concat([train_label, val_label, test_label], axis=0)

        return split_data, split_label

    def get_training_dataset(self):
        """
        Returns the training dataset.

        :return: A Pandas DataFrame containing the training dataset.
        """
        split_data, split_label = self.get_splitter()
        return split_data.iloc[:int((1-self.test_size-self.validation_size)*len(self.data))], split_label.iloc[:int((1-self.test_size-self.validation_size)*len(self.data))]

    def get_testing_dataset(self):
        """
        Returns the testing dataset.

        :return: A Pandas DataFrame containing the testing dataset.
        """
        split_data, split_label = self.get_splitter()
        return split_data.iloc[int((1-self.test_size)*len(self.data)):], split_label.iloc[int((1-self.test_size)*len(self.data)):]

    def get_validation_dataset(self):
        """
        Returns the validation dataset.

        :return: A Pandas DataFrame containing the validation dataset.
        """
        split_data, split_label = self.get_splitter()
        return split_data.iloc[int((1-self.test_size-self.validation_size)*len(self.data)):int((1-self.test_size)*len(self.data))], split_label.iloc[int((1-self.test_size-self.validation_size)*len(self.data)):int((1-self.test_size)*len(self.data))]

