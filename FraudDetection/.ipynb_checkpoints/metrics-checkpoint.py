import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import csv

class Metrics:
    def __init__(self):
        self.y_prediction = None
        self.y_label = None
        file_path = os.path.join('results', 'report.csv')
        if os.path.exists(file_path):
            os.remove(file_path)

    def precision(self):
        """
        Calculate precision score.
        """
        return precision_score(self.y_label, self.y_prediction)
#        return precision_score(self.y_label, self.y_prediction, average='weighted')


    def recall(self):
        """
        Calculate recall score.
        """
        return recall_score(self.y_label, self.y_prediction)
#        return recall_score(self.y_label, self.y_prediction, average='weighted')


    def sensitivity(self):
        """
        Calculate sensitivity (recall) score.
        """
        cm = confusion_matrix(self.y_label, self.y_prediction)
        return cm[1,1] / (cm[1,0] + cm[1,1])

    def specificity(self):
        """
        Calculate specificity score.
        """
        cm = confusion_matrix(self.y_label, self.y_prediction)
        return cm[0,0] / (cm[0,0] + cm[0,1])

    def f1_score(self):
        """
        Calculate F1 score.
        """
        #return f1_score(self.y_label, self.y_prediction, average='weighted')
        return f1_score(self.y_label, self.y_prediction)

    def roc_auc_score(self):
        """
        Calculate ROC AUC score.
        """
        #return roc_auc_score(self.y_label, self.y_prediction, average='weighted', multi_class='ovr')
        return roc_auc_score(self.y_label, self.y_prediction)

    def accuracy_score(self):
        """
        Calculate accuracy score.
        """
        return accuracy_score(self.y_label, self.y_prediction)

    def run(self, y_prediction, y_label, fold, testing):
        """
        Generate a report with metrics.
        """
        self.y_prediction = y_prediction
        self.y_label = y_label

        report = [
            ('Metric', 'Value'),
            ('Precision', self.precision()),
            ('Recall', self.recall()),
            ('Sensitivity', self.sensitivity()),
            ('Specificity', self.specificity()),
            ('F1 Score', self.f1_score()),
            ('ROC AUC Score', self.roc_auc_score()),
            ('Accuracy Score', self.accuracy_score())
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