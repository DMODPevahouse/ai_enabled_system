# For Fraud detection: 
The purpose of this is to take in information of fraud and non fraud transactions, train on that data, and predict whether or not that data is fraud or not fraud. The motivation is to make sure a model can accurately predict if there is currently fraud to let a user know. The below command is how to retrieve the docker
### docker run --restart=unless-stopped -it -p 8788:8788 -v < your workspace here >:/workspace dmodpevahouse/705.603spring24:assignment5_1
There are a few commands that can be run after this image is started. 
### http://localhost:8787/stats -- this tells the user that to get stats to run:
### http://localhost:8787/crossvalidate -- runs a 5 fold cross validation that will take around 2 hours to run, depending on the system and the data file
### http://localhost:8787/infer?merchant=fraud_Lind&category=misc_net&amt=5.12&first=Tyler&last=Banks&sex=Male&lat=55.4&long=44.3&city_pop=4000&job=IT&merch_lat=33.2&merch_long=22.3&day_of_week=1&day_of_month=5&time=12&generation=4
## Definition of the terms and what it is expecting 
first -- string \
last -- string \
sex -- string \
lat -- float \
long -- float \
city_pop -- int \
job -- string \
merch_lat -- float \
merch_long -- float \
day_of_week -- int 1-7 for Monday-Sunday\
day_of_month -- int 1-31 \
generation -- int 1-6, where 1 is 1901-1928, 2 is 1928-1946, 3 is 1945-1965, 4 is 1965-1981, 5 is 1981-1997, 6 is 1997-2013 \
This will give an inference based on the values changing
### http://localhost:8787/predict -- this is a post that allows you to send a new transactions.csv to train the data -- have not had the time to fully test this feature so may not be completely done
### http://localhost:8787/backup -- if you have replaced your data with a new transaction, this returns it to the original -- also not very tested (WIP)
### http://localhost:8787/predict -- allows you to send a json file to get an inference instead of the infer method above