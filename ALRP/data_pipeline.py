import os
import cv2
import numpy as np
from model import ObjectDetection
from models_nms import non_max_suppression
import imutils
import ffmpeg

class LicensePlateETL:
    """
    A class for extracting, transforming, and loading license plate images from a video.
    """
    def __init__(self, image_directory='images', output_directory='output'):
        """
        Initialize the LicensePlateETL class.

        :param image_directory: The directory containing the extracted images. Default is 'images'.
        :param output_directory: The directory where the cropped license plate images will be saved. Default is 'output'.
        """
        self.image_directory = image_directory
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
    
    
    def extract(self, input_url, width, height, frames_per_second):
        """
        Extract frames from a video and save them as images.

        :param input_url: The URL or file path of the input video.
        :param width: The width of the frames to extract.
        :param height: The height of the frames to extract.
        :param frames_per_second: The number of frames to extract per second.
        """
        frame_count = 0
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
        process1 = (
            ffmpeg
            .input(input_url)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        while True:
            in_bytes = process1.stdout.read(width * height * 3)
            if not in_bytes:
                break
            in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

            if frame_count % (frames_per_second) == 0:
                cv2.imwrite(os.path.join(self.image_directory, f'frame_{frame_count}.jpg'), in_frame)


            frame_count += 1     
        print("video accepted")
        process1.wait()
    

    def transform(self, weights, config):
        """
        Perform object detection on the extracted images and crop the images to contain only the license plates.

        :param weights: The path to the object detection model weights.
        :param config: The path to the object detection model configuration file.
        
        The goal is to have a directory of images cropped and distorted read for OCR after this is run
        """
        od = ObjectDetection(weights, config)

        # Loop over all the images in the directory
        for filename in os.listdir(self.image_directory):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                # Load the image
                image_path = os.path.join(self.image_directory, filename)
                image = cv2.imread(image_path )
                image = cv2.resize(image, (620,480) )
                # Perform object detection
                objects = od.detect(image_path)
                box_list = []
                confidence_list = []
                if objects != []:
                    for box in objects:
                        if box != []: 
                            box_list.append(box['box'])
                            confidence_list.append(box['confidence'])
                    keep = non_max_suppression(box_list, confidence_list, .5)

                    # Crop the image based on the detected object's bounding box
                    # Get the first detected object
                    obj = objects[keep[0]]

                    # Get the bounding box coordinates
                    x, y, w, h = obj["box"]

                    # Crop the image
                    cropped_image = image[y:y+h, x:x+w]

                    # Save the cropped image
                    if cropped_image is None or cropped_image.size != 0:
                        self.load(filename, cropped_image)
                
    
    def load(self, filename, cropped_image):
        """
        Save the cropped image to the output directory.

        :param filename: The name of the original image file.
        :param cropped_image: The cropped image to be saved.
        """
        if cropped_image is None or cropped_image.size == 0:
            raise ValueError("Cropped image is empty or None.")
        #cropped_image = cv2.resize(cropped_image, (620,480) )
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 65, 65)
        #edged = cv2.Canny(gray, 150, 200)
        cropped_image_path = os.path.join(self.output_directory, f"cropped_{filename}")
        cv2.imwrite(cropped_image_path, gray)
