#FROM python:3.9
# Install necessary packages
#RUN apt-get update && apt-get install -y \
#    build-essential \
#    libgl1-mesa-glx \
#    libegl1-mesa \
#    xserver-xorg \
#    x11-xserver-utils \
#    libxi-dev \
#    libxcursor-dev \
#    libxrandr-dev \
#    libxinerama-dev \
#    libxcomposite-dev \
#    libxdamage-dev \
#    libxfixes-dev \
#    libxrandr-dev \
#    libxml2-dev \
#    libxext-dev \
#    libxmu-dev \
#    libxpm-dev \
#    mesa-common-dev \
#    xvfb \
#    wget
#COPY . ./
#RUN pip3 install -r requirements.txt
#CMD ["deployment_udp_client.py"]
#ENTRYPOINT ["python"]
FROM python:3.11.7
 
RUN apt update
RUN apt-get update -o Acquire::Check-Valid-Urls=0
RUN apt install -y ffmpeg
RUN apt install -y tesseract-ocr 
COPY . ./

RUN pip install ffmpeg-python opencv-python-headless opencv-python
RUN pip3 install -r requirements.txt
 

 
CMD ["alrp_service.py"]
ENTRYPOINT [ "python" ]