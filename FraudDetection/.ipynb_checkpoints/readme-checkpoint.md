# For Fraud detection: 
The purpose of this is to take in information of fraud and non fraud transactions, train on that data, and predict whether or not that data is fraud or not fraud. The motivation is to make sure a model can accurately predict if there is currently fraud to let a user know. The below command is how to retrieve the docker
### docker run --restart=unless-stopped -it -p 8788:8788 -v < your workspace here >:/workspace dmodpevahouse/705.603spring24:assignment5_1
There are a few commands that can be run after this image is started. 
### http://localhost:8788/predict -- allows you to send a json file to get an inference instead of the infer method above, an example json file is in /workspace, feel free to adjust amounts but stay within the same datatype and it will predict fraud vs nonfraud
### http://localhost:8788/stats -- this tells the user that to get stats to run:
### http://localhost:8788/crossvalidate -- runs a 5 fold cross validation that will take around 2 hours to run, depending on the system and the data file
### http://localhost:8788/infer?merchant=fraud_Lind&category=misc_net&amt=5.12&first=Tyler&last=Banks&sex=Male&lat=55.4&long=44.3&city_pop=4000&job=IT&merch_lat=33.2&merch_long=22.3&day_of_week=1&day_of_month=5&time=12&generation=4 -- this is WIP and will not work
### http://localhost:8788/backup -- if you have replaced your data with a new transaction, this returns it to the original -- also not very tested