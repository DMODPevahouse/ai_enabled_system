import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

class ObjectDetection:
    def __init__(self, weights, config):
        # Load YOLOv3 model configuration and weights
        self.config_file = config
        self.weight_file = weights

        # Load the network
        self.net = cv2.dnn.readNet(self.config_file, self.weight_file)

        # Define the list of classes
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Get the names of the output layers
        self.output_layers = self.net.getUnconnectedOutLayersNames()

    def detect(self, img):
        # Get the height and width of the input image
        image = cv2.imread(img)
        image = cv2.resize(image, (620,480) )
        height, width, _ = image.shape
        confThreshold = .5
        # Create a blob from the input image
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), True, crop=False)

        # Set the input for the YOLOv3 model
        self.net.setInput(blob)

        # Get the output from the YOLOv3 model
        outs = self.net.forward(self.output_layers)

        # Initialize the list of detected objects
        objects = []
        frameHeight = image.shape[0]
        frameWidth = image.shape[1]
        # Loop over the output layers
        for out in outs:
            for detection in out:
                # if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)
                # if scores[classId]>confThreshold:
                confidence = scores[classId]
                if confidence > confThreshold:
                    class_id = np.argmax(detection[5:])
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_name = self.classes[class_id]
                    objects.append({
                        "class_name": class_name,
                        "image_name": img,
                        "confidence": float(confidence),
                        "box": np.array([left, top, width, height])
                    })
        if objects is not None:
            return objects

        # Return the list of detected objects
        objects.append({
                        "class_name": "no object type with confidence over 50",
                        "confidence": 0,
                        "box": 0
                    })
        # Return the list of detected objects
        return objects
    
    

def ocr_license_plate(image):
    """
    Extracts the license plate number using Google's Tesseract OCR.

    Parameters:
    image (PIL.Image): The cropped image.

    Returns:
    str: The extracted license plate number.
    """

    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    text = pytesseract.image_to_string(image, lang='eng', config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return text.strip()