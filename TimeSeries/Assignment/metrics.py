import os
import numpy as np
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv

class Metrics:
    def __init__(self, file_name):
        """
        This class if defined to create a report based off of a list of metrics to evaluate the model
        """
        self.y_prediction = None
        self.y_label = None
        self.file_name = file_name
        file_path = os.path.join('results', self.file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

    def mean_squared_error(self):
        """
        Calculate MSE.
        """
        return mean_squared_error(self.y_label, self.y_prediction)

    def rmse(self):
        """
        Calculate RMSE.
        """
        return rmse(self.y_label, self.y_prediction)

    def mean_absolute_error(self):
        """
        Calculate MAE.
        """
        return mean_absolute_error(self.y_label, self.y_prediction)

    def run(self, y_prediction, y_label):
        """
        Generate a report with metrics.
        :param y_prediction: the predictions the model has made
        :param y_labels: the actual labels of the data to be tested
        :param fold: the current fold to be mentioned in the report for tracking
        :param testing: if the data is training, testing, or validation
        """
        self.y_prediction = y_prediction["transaction_count"].to_numpy()
        self.y_label = y_label["transaction_count"].to_numpy()

        report_count = [
            ('Metric', 'Value'),
            ('mean_squared_error', self.mean_squared_error()),
            ('rmse', self.rmse()),
            ('mean_absolute_error', self.mean_absolute_error()),
        ]
        self.y_prediction = y_prediction["fraud_count"].to_numpy()
        self.y_label = y_label["fraud_count"].to_numpy()
        report_fraud = [
            ('Metric', 'Value'),
            ('mean_squared_error', self.mean_squared_error()),
            ('rmse', self.rmse()),
            ('mean_absolute_error', self.mean_absolute_error()),
        ]

        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')

        # Save report to a csv file
        with open(os.path.join('results', self.file_name), 'a', newline='') as f:
            writer = csv.writer(f)
            title = "count"
            writer.writerows(title)
            writer.writerows(report_count)
            title = "fraud"
            writer.writerows(title)
            writer.writerows(report_fraud)

       