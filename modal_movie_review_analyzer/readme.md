# For Movie Review Placement: 
The purpose of this docker container is to take a json file, example given in the container, of data that includes a review of a specified movie. Once that is done the system will give a rating of the movie based on the sentiment of the description. The sentiments will still range from 1-5 with one being negative, and 5 being the best review, but the system itself will give that rating, instead of a person to make the rating system mroe consistent. The below command is how to retrieve the docker
### docker run --restart=unless-stopped -it -p 8792:8792 -v < your workspace here >:/workspace dmodpevahouse/705.603spring24:assignment12_1
There are a few commands that can be run after this image is started. 
### http://localhost:8788/predict -- allows you to send a json file to get an inference instead of the infer method above, an example json file is in /workspace, feel free to adjust amounts but stay within the same datatype and it will give a raiting based off of the sentiment of the review
### http://localhost:8788/stats -- this tells the user that to get stats to run:
### http://localhost:8788/crossvalidate -- runs a 5 fold cross validation that will take around 2 hours to run, depending on the system and the data file
