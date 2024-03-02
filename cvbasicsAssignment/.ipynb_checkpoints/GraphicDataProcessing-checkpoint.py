import cv2
import numpy as np

class ObjectDetection:
    def __init__(self):
        # Load YOLOv3 model configuration and weights
        self.config_file = "yolov3.cfg"
        self.weight_file = "yolov3.weights"

        # Load the network
        self.net = cv2.dnn.readNet(self.config_file, self.weight_file)

        # Define the list of classes
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Get the names of the output layers
        self.output_layers = self.net.getUnconnectedOutLayersNames()

    def detect(self, image):
        # Get the height and width of the input image
        image = cv2.imread("image.jpg")
        height, width, _ = image.shape

        # Create a blob from the input image
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), True, crop=False)

        # Set the input for the YOLOv3 model
        self.net.setInput(blob)

        # Get the output from the YOLOv3 model
        outs = self.net.forward(self.output_layers)

        # Initialize the list of detected objects
        objects = []

        # Loop over the output layers
        for out in outs:
            # Loop over the detections
            for detection in out:
                # Get the confidence of the detection
                confidence = detection[4]

                # Filter detections with confidence below a threshold
                if confidence > 0.5:
                    # Get the index of the class with the highest confidence
                    class_id = np.argmax(detection[5:])

                    # Get the name of the class
                    class_name = self.classes[class_id]

                    # Get the coordinates of the bounding box
                    box = detection[0:4] * np.array([width, height, width, height])

                    # Append the detected object to the list
                    objects.append({
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "box": box.astype("int")
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