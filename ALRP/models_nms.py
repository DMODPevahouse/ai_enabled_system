import numpy as np

def non_max_suppression(boxes, scores, threshold):
    """
    Perform non-maximal suppression to eliminate redundant overlapping bounding boxes.
    
    Parameters:
    - boxes: List of bounding boxes in format [x1, y1, x2, y2].
    - scores: Confidence scores for each bounding box.
    - threshold: IoU threshold for determining when to eliminate boxes.
    
    Returns:
    - List of indices of bounding boxes that were kept.
    """
    # Convert to numpy arrays for easier mathematical operations
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores)

    # Initialize a list to keep the indices of the boxes we decide to keep
    keep = []

    # Extract the coordinates of all boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Sort the indices of the boxes by score, in ascending order (lowest confidence first)
    idxs = np.argsort(scores)

    # Loop through the boxes, starting with the highest score
    while len(idxs) > 0:
        # The index of the current box with the highest score
        last = len(idxs) - 1
        i = idxs[last]
        # Add the index of the box with the highest score to the list of boxes to keep
        keep.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y)
        # coordinates for the end of the bounding box among the remaining boxes, to calculate overlap
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the overlapping area, and hence the area
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        # Delete indexes from the list that have an overlap greater than the threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return keep


def iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - boxA: The first bounding box as a list of coordinates [x1, y1, x2, y2].
    - boxB: The second bounding box as a list of coordinates [x1, y1, x2, y2].

    Returns:
    - The IoU as a float.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou