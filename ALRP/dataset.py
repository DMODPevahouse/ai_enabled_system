import os
import random
import numpy as np

class Object_Detection_Dataset:
    """
    ImageDataset is a class that represents a dataset of images. It provides methods to split the dataset into training, testing, and validation sets.

    Attributes:
        image_dir (str): The directory where the images are stored.
        num_folds (int): The number of folds to split the dataset into.
        image_files (list): A list of the image files in the dataset.
        num_images (int): The total number of images in the dataset.

    Methods: __init__(self, image_dir, num_folds=5): Initializes the ImageDataset object with the given image directory and number of folds.
        split_train_test_valid(self, fold): Splits the dataset into training, testing, and validation sets based on the given fold.
        get_train_data(self, fold): Returns the training data for the given fold.
        get_test_data(self, fold): Returns the testing data for the given fold.
        get_valid_data(self, fold): Returns the validation data for the given fold.
    """
    def __init__(self, image_dir, num_folds=5):
        """
        Initializes the ImageDataset object with the given image directory and number of folds.

        Args:
            image_dir (str): The directory where the images are stored.
            num_folds (int): The number of folds to split the dataset into. Default is 5.

        Attributes:
            image_dir (str): The directory where the images are stored.
            num_folds (int): The number of folds to split the dataset into.
            image_files (list): A list of the image files in the dataset.
            num_images (int): The total number of images in the dataset.
        """
        self.image_dir = image_dir
        self.num_folds = num_folds
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(self.image_files)
        self.num_images = len(self.image_files)
        self.images_per_fold = self.num_images // self.num_folds

    def split_train_test_valid(self, fold):
        """
        Splits the dataset into training, testing, and validation sets based on the given fold.
        Args: fold (int): The fold number to split the dataset into.
        Returns: tuple: A tuple containing three lists: train_files, test_files, and valid_files.
        Raises: ValueError: If the fold number is less than 0 or greater than the number of folds.
        """
        if fold < 0 or fold >= self.num_folds:
            raise ValueError("fold must be between 0 and {}".format(self.num_folds))
        train_files = self.image_files[:fold*self.images_per_fold]
        test_files = self.image_files[(fold*self.images_per_fold):(fold*self.images_per_fold) + self.images_per_fold]
        valid_files = self.image_files[(fold*self.images_per_fold) + self.images_per_fold:]
        return train_files, test_files, valid_files

    def get_training_dataset(self, fold):
        """
        Returns the training data for the given fold.
        Args: fold (int): The fold number to retrieve the training data for.
        Returns: list: A list of strings representing the file paths of the training images.
        Raises: ValueError: If the fold number is less than 0 or greater than the number of folds.
        """
        train_files, _, _ = self.split_train_test_valid(fold)
        return [os.path.join(self.image_dir, f) for f in train_files]

    def get_testing_dataset(self, fold):
        """
        Returns the testing data for the given fold.
        Args: fold (int): The fold number to retrieve the testing data for.
        Returns: list: A list of strings representing the file paths of the testing images.
        Raises: ValueError: If the fold number is less than 0 or greater than the number of folds.
        """
        _, test_files, _ = self.split_train_test_valid(fold)
        return [os.path.join(self.image_dir, f) for f in test_files]

    def get_validation_dataset(self, fold):
        """
        Returns the validation data for the given fold.
        Args: fold (int): The fold number to retrieve the validation data for.
        Returns: list: A list of strings representing the file paths of the validation images.
        Raises: ValueError: If the fold number is less than 0 or greater than the number of folds.
        """
        _, _, valid_files = self.split_train_test_valid(fold)
        return [os.path.join(self.image_dir, f) for f in valid_files]