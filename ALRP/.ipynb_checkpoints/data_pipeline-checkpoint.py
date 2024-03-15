import os
import cv2
import numpy as np
from model import ObjectDetection
from models_nms import non_max_suppression

class LicensePlateETL:
    def __init__(self):
        self.none = None
    
    
    def extract(self, input_url, width, height, frames_per_second):
        """
        Stream video from a given input URL using ffmpeg and display it with OpenCV.

        This function opens a video stream from the specified input URL, decodes it using
        ffmpeg to raw video frames, and displays these frames using OpenCV. The function
        continues streaming until the video feed ends or is manually terminated.

        Parameters:
        - input_url : str
            The URL of the video stream to open. This can be any valid ffmpeg input, 
            such as a file path, RTSP, or UDP stream URL.
        - width : int
            The width of the video frames to be displayed.
        - height : int
            The height of the video frames to be displayed.

        Note:
        - To exit the video stream display, press 'q' while the OpenCV window is focused.

        Take this code, and create an extract function in the ETL with it by converting it to just create images from the code instead of just showing it
        """
        frame_count = 0
        images_directory = 'output'
        if not os.path.exists(images_directory):
            os.makedirs(images_directory)

        cv2.namedWindow("Video Stream")

        process1 = (
            ffmpeg
            .input(input_url)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        while True:
            print("tesT")
            in_bytes = process1.stdout.read(width * height * 3)
            if not in_bytes:
                break
            print("here")
            in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            cv2.imshow("Video Stream", in_frame)

            if frame_count % (frames_per_second) == 0:
                cv2.imwrite(os.path.join(images_directory, f'frame_{frame_count}.png'), in_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

            frame_count += 1

        process1.wait()
        cv2.destroyAllWindows()
    

    def transform(self, image_directory, weights, config):
        od = ObjectDetection(weights, config)

        # Loop over all the images in the directory
        for filename in os.listdir(image_directory):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                # Load the image
                image_path = os.path.join(image_directory, filename)
                image = cv2.imread(image_path )

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
                    w, h, x, y = obj["box"]

                    # Crop the image
                    cropped_image = image[y:y+h, x:x+w]

                    # Save the cropped image
                    self.load('output', filename, cropped_image)
                
    
    def load(self, image_directory, filename, cropped_image):
        cropped_image_path = os.path.join(image_directory, f"cropped_{filename}")
        cv2.imwrite(cropped_image_path, cropped_image)

