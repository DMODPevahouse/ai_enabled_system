import os
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
import csv

class Metrics:
    def __init__(self):
        """
        This class if defined to create a report based off of a list of metrics to evaluate the model
        """
        self.y_prediction = None
        self.y_label = None
        file_path = os.path.join('results', 'report.csv')
        if os.path.exists(file_path):
            os.remove(file_path)

    def reciprocal_rank(self, k=2):
        """
        Calculate the Mean Reciprocal Rank (MRR) for the given true labels and predicted scores.

        Parameters:
            y_true (array-like): The true labels for the test data.
            y_score (array-like): The predicted scores for the test data.
            k (int): The number of top predictions to consider.

        Returns:
            float: The Mean Reciprocal Rank (MRR) for the given true labels and predicted scores.
        """
        k=k
        ranks = np.zeros(len(self.y_label_rr))
        for i in range(len(self.y_label_rr)):
            scores = self.y_prediction_rr[i]
            labels = self.y_label_rr[i]
            sorted_indices = np.argsort(-scores)
            for j in range(k):
                if self.y_label_rr[sorted_indices[j]] == labels:
                    ranks[i] = 1 / (j + 1)
                    break
        return np.mean(ranks)

    def average_precision_score(self):
        """
        Calculate average precision score.
        """
        return average_precision_score(self.y_label_map, self.y_prediction_map, average='macro', pos_label=max(self.y_label_map))

    def accuracy_score(self):
        """
        Calculate accuracy score.
        """
        return accuracy_score(self.y_label, self.y_prediction)
    
    def run(self, y_prediction, y_label, fold, testing):
        """
        Generate a report with metrics.
        :param y_prediction: the predictions the model has made
        :param y_labels: the actual labels of the data to be tested
        :param fold: the current fold to be mentioned in the report for tracking
        :param testing: if the data is training, testing, or validation
        """
        k=2
        top_k_indices = np.argsort(y_prediction, axis=1)[:, -k:]
        top_k_labels = np.array([y_label.iloc[i] for i in top_k_indices.ravel()])
        top_k_indices_test = np.searchsorted(np.cumsum(np.bincount(top_k_indices.ravel())), np.arange(len(top_k_indices)))
        top_k_scores = y_prediction[np.arange(len(y_label))[:, np.newaxis], top_k_indices]
        test_indices = np.searchsorted(np.argsort(y_label), np.arange(len(y_label)))
        
        self.y_prediction_map = top_k_scores.ravel()[test_indices]
        self.y_label_map = top_k_labels[test_indices]
        self.y_prediction_rr = y_prediction
        self.y_label_rr = y_label.to_numpy()
        report = [
            ('Metric', 'Value'),
            ('Average Precision Score', self.average_precision_score()),
            ('Mean Reciprocal Rank', self.reciprocal_rank()),
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