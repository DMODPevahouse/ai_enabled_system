import numpy as np
import cv2
import ffmpeg
import os
import subprocess

def stream_video(input_url, width, height, frames_per_second):
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
    """
    frame_count = 0
    images_directory = 'output_testing'
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)
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

        if frame_count % (frames_per_second) == 0:
            cv2.imwrite(os.path.join(images_directory, f'frame_{frame_count}.png'), in_frame)


        frame_count += 1
        #cv2.imshow("Video Stream", in_frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        #    break
    print("video accepted")
    process1.wait()
    #cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    in_file = 'udp://127.0.0.1:23000'  # Example UDP input URL
    width = 3840  # Example width
    height = 2160  # Example height
    stream_video(in_file, width, height, 10)