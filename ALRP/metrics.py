import matplotlib.pyplot as plt
import cv2
import pytesseract
from PIL import Image
import numpy as np
import os
import model
from dataset import Object_Detection_Dataset
import json
import difflib
from metrics_mAP import calculate_precision_recall_curve, calculate_map_11_point_interpolated
import csv

class Metrics:
    """
    Class for computing various metrics related to image classification.

    Attributes:
        image_directory (str): Path to the directory containing the images.
        json_file (str): Path to the JSON file containing the answers.
        k (int): Number of top similar images to consider for similarity metric.
    """
    def __init__(self, image_directory, answers, k=5):
         """
        Initialize the Metrics class.

        Args:
            image_directory (str): Path to the directory containing the images.
            json_file (str): Path to the JSON file containing the answers.
            k (int, optional): Number of top similar images to consider. Defaults to 5.
        """
        self.image_directory = image_directory
        self.json_file = answers
        self.k = k
        
        
    def run_test(self, image_directory_list, data):
        """
        Run a test for image classification by computing string similarity scores.

        This method iterates over the given image directories, performs OCR on each image,
        and computes the string similarity score between the OCR output and the corresponding
        ground truth label.

        Args:
            image_directory_list (list): List of paths to the directories containing the images.
            data (dict): Dictionary mapping image paths to their corresponding ground truth labels.

        Returns:
            tuple: Tuple containing two numpy arrays:
                y_prediction_list (numpy.ndarray): Array of string similarity scores.
                y_label_list (numpy.ndarray): Array of binary labels (1 for correct classification).
        """
        y_prediction_list, y_label_list = [], []
        for image in image_directory_list:
            text = model.ocr_license_plate(image)
            label = data[image]
            y_prediction =  self.string_similarity_score(text, label)
            y_label = 1
            y_prediction_list.append(y_prediction)
            y_label_list.append(y_label)
        return np.array(y_prediction_list), np.array(y_label_list)
              
    def generate_report(self):
        """
        Generate a report by running tests on training, testing, and validation datasets.

        This method generates a report by splitting the dataset into training, testing,
        and validation sets, running tests on each set, and computing metrics for each set.
        """
        odd = Object_Detection_Dataset(self.image_directory)
        with open(self.json_file, 'r') as f:
            # Load the JSON data as a Python dictionary
            data = json.load(f)
        for i in range(self.k):
            training = odd.get_training_dataset(i)
            testing = odd.get_testing_dataset(i)
            validation = odd.get_validation_dataset(i)
            y_preds, y_labels = self.run_test(training, data)
            self.run(y_preds, y_labels, i, "training")
            y_preds, y_labels = self.run_test(testing, data)
            self.run(y_preds, y_labels, i, "testing")
            y_preds, y_labels = self.run_test(validation, data)
            self.run(y_preds, y_labels, i, "validation")
            
    def string_similarity_score(self, string1, string2):
        """
        Compute the string similarity score using the SequenceMatcher ratio.

        This method computes the string similarity score between two input strings using
        the `difflib.SequenceMatcher.ratio()` method.

        Args:
            string1 (str): The first input string.
            string2 (str): The second input string.

        Returns:
            float: The string similarity score between 0 and 1, where 1 indicates
            a perfect match.
        """
        matcher = difflib.SequenceMatcher(None, string1, string2)
        similarity = matcher.ratio()
        return similarity
    
    def run(self, y_prediction, y_label, fold, testing):
        """
        Generate a report with metrics.
        :param y_prediction: the predictions the model has made
        :param y_labels: the actual labels of the data to be tested
        :param fold: the current fold to be mentioned in the report for tracking
        :param testing: if the data is training, testing, or validation
        """
        precision, recall, thresholds = calculate_precision_recall_curve(y_label, y_prediction)

        precision_recall_points = zip(precision, recall)

        map_value = calculate_map_11_point_interpolated(precision_recall_points)

        
        report = [
            ('Metric', 'Value'),
            ("Prediction similarity Percentage", y_prediction),
            ('Precision', precision),
            ('Recall', recall),
            ('Mean Average Precision', map_value),
        ]

        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')

        # Save report to a csv file
        with open(os.path.join('results', 'report.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            iteration = [(testing, str(fold))]
            writer.writerows(iteration)
            writer.writerows(report)

        return report