# For Fraud Forcast (TimeSeries): 
The main purpose of this container is to take in a csv of fraud data, determine how much fraud happens on a given day and predicts what the next calendar year could have for fraud. With this container you will train a traditional and a deep model on the data, and have it predict a date in the next calendar year of how many transactions and fraud there would be on a given day
### docker run --restart=unless-stopped -it -p 8791:8791 -v < your workspace here >:/workspace dmodpevahouse/705.603spring24:assignment10_1
There are a few commands that can be run after this image is started. 
### http://localhost:8791/stats -- this tells the user that to get stats to run:
### http://localhost:8791/test -- trains on part of the data and tests against the rest. Could take 15-30 minutes to run but will report the stats in results/deep_report.csv and results/trad_report.csv
### http://localhost:8791/infer?mm-dd=02-15 -- Pass the model the mm-dd of the day within the next year you would like to predict. 02-15 could be any value of mm-dd format you would like it to be
