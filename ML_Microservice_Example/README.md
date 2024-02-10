# ML_Microservice_Example
Example code to build microservice to return learning stats and provide inference

Provides two microservices
1) returns performance stats - http://localhost:8786/stats
2) returns inference determination given an age and salary - http://localhost:8786/infer?age=45&salary=40000


The code for this assignment did not change, but still works successfully when pulled from the docker image using the command:

docker run --restart=unless-stopped -it -p 8786:8786 -v < your workspace here>:/rapids/notebooks/workspace dmodpevahouse/705.603spring24:assignment2_1

Which requires you to enter your given directory, though it does not use any daya from there so you do not need to pass the directory entirely, but it does help if you want to do any work on it. 

Changes:

changed sklearn to scikit-learn in the requirements.txt


removed os from the requirements.txt

## Design Outline in Design.ipynb
Populate throughout for your final project

