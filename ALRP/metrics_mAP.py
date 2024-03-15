import numpy as np


def calculate_precision_recall_curve(y_true, scores):
    """
    Manually calculate the precision-recall curve.

    Parameters:
    - y_true : array-like of shape (n_samples,)
               True binary labels in range {0, 1}.
    - scores : array-like of shape (n_samples,)
               Target scores, can either be probability estimates of the positive class,
               confidence values, or non-thresholded measure of decisions.

    Returns:
    - precision : list
                  Precision values such that element i is the precision of
                  predictions with score >= thresholds[i].
    - recall : list
               Recall values such that element i is the recall of
               predictions with score >= thresholds[i].
    - thresholds : list
                   Decreasing thresholds on the decision function used to compute
                   precision and recall.
    """
    # Sort scores and corresponding true values
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_true = y_true[sorted_indices]
    
    # Append an extra threshold to cover all cases
    thresholds = np.append(sorted_scores, [0], axis=0)
    precision = []
    recall = []
    TP = 0
    FP = 0
    FN = np.sum(y_true) # Initial false negatives are all positives
    
    for i in range(len(sorted_scores)):
        if sorted_true[i] == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
        
        prec = TP / (TP + FP) if TP + FP > 0 else 0
        rec = TP / (TP + FN) if TP + FN > 0 else 0
        
        precision.append(prec)
        recall.append(rec)
    
    # Ensure last point is at recall zero
    precision.append(1.0)
    recall.append(0.0)
    
    return precision, recall, thresholds[:-1]


def calculate_map_11_point_interpolated(precision_recall_points):
    """
    Calculate the mean average precision (mAP) using 11-point interpolation from a list of precision-recall points.
    
    Parameters:
    - precision_recall_points: A list of tuples, where each tuple represents a (recall, precision) point.
    
    Returns:
    - The mAP value as a float.
    """
    # Ensure the list is sorted by recall in ascending order
    precision_recall_points = sorted(precision_recall_points, key=lambda x: x[0])
    
    interpolated_precisions = []
    for recall_threshold in [i * 0.1 for i in range(11)]:
        # Find all precisions with recall greater than or equal to the threshold
        possible_precisions = [p for r, p in precision_recall_points if r >= recall_threshold]
        
        # Interpolate precision: take the maximum precision to the right of the current recall level
        if possible_precisions:
            interpolated_precisions.append(max(possible_precisions))
        else:
            interpolated_precisions.append(0)
    
    # Calculate the mean of the interpolated precisions
    mean_average_precision = sum(interpolated_precisions) / len(interpolated_precisions)
    
    return mean_average_precision


if __name__ == "__main__":

    # Example usage of calculate_map_11_point_interpolated():
    y_true = np.array([0, 1, 1, 0, 1]) # Actual labels
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7]) # Predicted scores/probabilities from object detection model.
    precision, recall, thresholds = calculate_precision_recall_curve(y_true, scores)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Thresholds:", thresholds)

    precision_recall_points = zip(precision, recall)

    map_value = calculate_map_11_point_interpolated(precision_recall_points)
    print(f"Mean Average Precision: {map_value:.4f}")
