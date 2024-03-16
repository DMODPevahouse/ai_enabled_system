# For automated License Plate Recognition: 
This docker containers purpose is to receive a video in batch manner, transform the image to prep it for being read by OCR, and give a csv of the results for investigation. To begin, run the below command
### docker run --restart=unless-stopped -it -p 8790:8790 -p 23000:23000/udp -v < your workspace here >:/workspace dmodpevahouse/705.603spring24:assignment8_1
Once the docker is built onto the your machine, it will show a message called "awaiting video" which would be the designated 4k video you want to try and predict license plate information from. Once that video is recieved, then it will go through and transform the jpgs created from the video, and attempt to get license plate numbers from that. To send a video, the command should look something like this:
## ffmpeg -i LicensePlateReaderSample_4k.mov -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
After that, it will take around 15 minutes to transform the images to ready them for being read, and then will be ready for the below commands
#
There are a few commands that can be run after this image is started. 
### http://localhost:8788/predict_tiny -- This will run an OCR on the images transformed from the video using the tiny weights and config. This will be in the workspace directory in a file called answers_tiny.csv
### http://localhost:8788/stats -- All this does is point you to model performance file in results directory tested on a test, training, and validation dataset
### http://localhost:8788/predict_normal -- This will run an OCR on the images transformed from the video using the normal weights and config. This will be in the workspace directory in a file called answers_normal.csv