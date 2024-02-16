# CarFactors
Example code to build microservice to return learning stats and provide inference of car factors. The motivation for this is to look at statistics and health of a car to predict when the next part failure will occur. 

Provides two microservices
1) returns performance stats - http://localhost:8787/stats
2) returns inference determination given an age and salary - http://localhost:8787/infer?transmission=mechanical&color=silver&odometer=10000&year=2010&bodytype=suv&price=10000


The code is what powers the below docker command, also available in dockerhub. Once the container is running, you can run the above commands. 

docker run --restart=unless-stopped -it -p 8787:8787 -v < your workspace here >:/rapids/notebooks/workspace dmodpevahouse/705.603spring24:assignment4_1

Which requires you to enter your given directory, though it does not use any data from there so you do not need to pass the directory entirely, but it does help if you want to do any work on the container. 
