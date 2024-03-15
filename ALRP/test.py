import subprocess
import sys
import cv2
import numpy
import pdb

def read_video_udp(host, port):
    """
    Reads in video over UDP using ffmpeg and Python.

    Args:
    host (str): The host address to receive the UDP stream.
    port (int): The port number to receive the UDP stream.

    Returns:
    None
    """
    pdb.set_trace()
    # Set up the ffmpeg command as a list of arguments
    command = ['ffmpeg',
               '-i', f'udp://{host}:{port}',
               '-f', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-']

    # Open a subprocess to execute the command
    pipe = subprocess.Pipe(stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    process = subprocess.Popen(command, stdin=pipe.stdin, stdout=pipe.stdout)

    # Read frames from the subprocess and display them
    while True:
        raw_frame = pipe.stdout.read(3 * 1920 * 1080)
        if not raw_frame:
            break

        # Convert the raw bytes to a numpy array
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((1080, 1920, 3))

        # Display the frame using OpenCV
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the subprocess and destroy the OpenCV window
    process.stdin.close()
    pipe.stdout.close()
    cv2.destroyAllWindows()
    
    
host = '127.0.0.1'
port = 23000

# Call the read_video_udp function to start receiving the UDP stream
read_video_udp(host, port)