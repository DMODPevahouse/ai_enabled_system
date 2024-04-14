# <u>MOTIVATION (Why's)</u>

## Why are we solving this problem?

* Problem Statement: This problem consists of trying to use previous data to determine what would be the best approach to forecast future transactions, both normal and fraudulent, and determine how many of said transactions would exist on a given date. This allows the business to forecast what the given effort will need for the future to determine the need

## Value Proposition

* Why is an ML solution a viable approach? 
Time series is a good solution for this type of problem as it takes in data that can happen over a period of time to see the rate of change and growth that happens over a period of time to apply that pattern to the future to see how that will interact. This gives events and correlations for growth and patterns for what happens over a given day, month, year, etc. 
* Is this problem suitable for machine learning due to its data and desired outcomes?
The problem is suitable for machine learning due to the fact that there are many years of data on the topic that seemingly follows a pattern that can be used and is useful to show how that can change in the future. Outside factors of course can change the data but overall there should be some sort of pattern to follow.
* Can we tolerate potential errors and uncertainties associated with ML?
Error is tolerable here, as mostly there is just a forecast of what could happen, so most likely there will be some sort of error in the prediction, and the main purpose it to give us a good idea of what could happen to prepare the business for that day if there was to be some sort of efficiencies based around patterns or preparations. 
* Is this solution technically feasible with current resources and technologies?
The solution here is feasible, as there are plenty of time series models, traditional and deep, that can solve this problem

# <u>REQUIREMENTS (What's)</u>

## SCOPE

* What are the overall goals and objectives of this project?
The end goal of this project is to have a docker container that has a web server with a given port, allowing people to send a specific date to the web server and receive the predicted transactions, normal and fraud, for that given date
* How will success be measured for this project?
Since we are predicting the future, the best way to measure success of the model would be, while testing, feed the model only part of the data, maybe 4 out of 5 years. Then have it predict the 4rth year, and measure the change against it. That will allow us to see the given pattern of that data. After that, when it comes to predicting the future we will have a basis on how much the deviation from the actual it could be.

## REQUIREMENTS

* **Assumptions:** We must have a decent amount, preferable several years, of data to establish a baseline and a growth line in order to accurately predict what will happen. Along with this, the assumption is that a specific date will be passed in the correct format to return an estimate.
* **System Requirements:** Requirements for the system is a docker container capable of creating a port that can be accessed from a browser, with the ability to receive a data to predict. There is also a need for data to train the model, which means we need a process to take in data, extract, load, and transform the data. There also needs to make to that the prediction only uses dates and not use others features. 

* What are potential risks and harms associated with this project?
The risks and harms of this project are relatively low, though not 0. While we are predicting the future so there is no real ground truth until we get there, if the errors are egregious, the business could be poorly prepared for what would happen. However, besides that, there are no big risks as this is moreso just to help prepare the business for that date. The business needs to be aware of potential risks and how far off we can predict the prediction will be so that the deviation could be accounted for as best as possible. There is also the possibility that there are external factors, lets say a market crash or a depression, that hampers transactions and drastically lowers them that could mean the model is predicting completely incorrect.
* What are the potential causes of errors or mistakes in the ML system?
There are risks that the system is really just forecasting events, and nothing can truly predict the future. Other external events like depressions, a pandemic, a massive trend on tik tok, that will affect transactions to a level that is outside of the bounds the model predicted, meaning lots of errors and risks of mistakes in the prediction. This could be alleviated by constantly adding new training data to the model, an online model, as it goes but that will not hide every issue. 

# <u>IMPLEMENTATION (How's)</u>

## DEVELOPMENT

* **Methodology:** There are several methodologies that will need to be considered when developing this algorithm. They will be listed below for explaining each part. 
Data Engineering Pipeline: this class shall take in the data, transform it, and then load it into a format that will be used in the algorithm. Otherwise known as the ETL, extract, transform, and load, method. The expectation is that we will take the data from previous transactions, limit to one, and possibly two for testing, features that will be used to build a model on how the pattern will be collected to forecast predictions. 
#
Dataset Partitioning: This class will take in a video format that will be transform the data into a feature or two to build the time series model. 
# 
Metrics: Since the point of the model is predicting the future, the best we can do for metrics is by taking a subset of the data, 4 out of 5 years for example, predict last year, compare the trends and determine the mean squared error or other metric for creating a deviation, so that when the model is released we can show the predicted deviation from the truth of what could happen on that day. 


***WIP***
* Mean Squared Error -- One good metric that will be a heavy focus for testing performance will be mean squared error, to show how far away the model is, and larger values will be more punishing. This will help determine the error that the model is and exagerates more as the model is, well, more inaccurate.
* Mean error -- Similar to MSE, just not squaring it to see what the mean error would be without squaring. 
#
* **High-level System Design:** The high level system design for this project is to have a set of data that the ETL will extract, transform to a few features that have the highest prediction capability, load it for the model to pull and use. Once that is finished, the model will be trained on the data that has been extracted a loaded, then it will go through testing which means that it will train, again, on a smaller subset of the data in order to predict on the data that it has not trained on. This could be a month, a year, etc. The purpose of this is to give some sort of deviation capability to see how the model will perform on unseen data. After that, training on the whole data will be needed. Once the final training has been done, the model will be set up in a docker container so that a user can use a web url connected to a specific port to pass a date. That date then be received and forecasted on the model with an explanation of the deviation previously found if possible.
* 
* **Development Workflow:** The development process will include taking the data, and investigating to determing what would be the best features for accurately predicting what is possible. 
## POLICY

* **Human-Machine Interfaces:** This model mostly has the humans interact with the final result. When the predictions occur, it gives a good metric to determine how the business should react in the future. Perhaps on weekends or holidays systems need to be prepared for surges, or maybe the month of May is very slow and to be prepared to slow things down, etc
* **Regulations:** Regulations here really determines on how the data is used. For the most part, since this is just an arbitrary prediction, the regulations itself may be pretty light.

## OPERATIONS

* **Continuous Deployment:** The CD/CI deployment would go like this: take data -> transform data -> use partial data to train -> test -> full train -> test on known data just to validate -> release and wait for dates to predict!
* **Post-deployment Monitoring:** Post deployment, the model and experts should compare on how accurate the model is being, see if any features would be benifical, and make sure there are no external factors drastically changing the ground truth vs the prediction, in essence what for slide
* **Maintenance:** Outlined slightly in the CD/CI, but seeing when the predictions start sliding away from the ground truth drastically, updating the model with current data to prevent said slide.
* **Quality Assurance:** Quality assurance will be determined by monitoring the data, predictions, and live ground truth to see if the model starts performing worse out of nowhere from external factors, and updating to account for slide as quickly as possible.

---
