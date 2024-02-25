# <u>MOTIVATION (Why's)</u>

## Why are we solving this problem?

* Problem Statement: The problem being faced in this situation is that currently an older ML model has started to catch fraudulent credit card reports. This is causing problems as customers are facing more and more fraud cases. So the goal would be to create a model that has a significant improvement over the older and depricating model. Without doing so, customer satisifaction and profitability starts to decline significantly.  

## Value Proposition

* Why is an ML solution a viable approach? 
Machine learning is an appropriate response to this problem for the fact that most fraud has some significant data that correlates with each other that is incredibly hard and time consuming for a human to catch quickly. With the data correlation built into a system, fraud can be got consistently with low risk and high reward. Fraud data is often incredibly large amounts of data, but smaller amounts of fraud that are hard to detect without pattern recognition.
* Is this problem suitable for machine learning due to its data and desired outcomes?
The problem is definitely suitable for machine learning, due to the fact that it is patterns being studied and recognized in order to predict fraud, and the amount of data available to the system would be substantial since the company is a banking company.
* Can we tolerate potential errors and uncertainties associated with ML?
It depends. False positive fraud detection is normally low risk and more of an annoyance then a problem, however false negative can have some drastic results if fraud is being ignored. This would cause significant loss if it is ignored, which it would be. That means that with this system, it needs to favor false positives more so then false negatives. Though it cannot call everything fraud as the more times a customer has to deal with that, customer satisfaction starts to heavily decrease which is also unacceptable. 
* Is this solution technically feasible with current resources and technologies?
The solution is feasible with current libraries like pytorch or scikit-learn, depending on the approach. Since the company is a bank with access to a large amount of data around fraud, that is also not a problem.

# <u>REQUIREMENTS (What's)</u>

## SCOPE

* What are the overall goals and objectives of this project?
The current solution has a precision rating of 40% and recall score of 70% which is unacceptable, leading to a loss of profitability and  customer satisfaction. The overal goal is the create a solution that will vastly improve those scores into an acceptable and meaningful manner. 
* How will success be measured for this project?
Success will be based on being  compared to the current solution, and beating the rating of 40% precision, and 70% recall. That means the values that are going to be investigated is the amount of true positives over true positives and false negatives, as well as true positives over true positives and false positives for the quantitative improvement data. 

## REQUIREMENTS

* **Assumptions:** Some key assumptions being attended to is that the data available to be trained on is well labeled and includes a wide variety of fraud attempts that a model could be trained on in order to detect fraud in a manner that is modern. Without this data, increasing the capability will not be possible. In addition, assumptions that there is time to develop and build the model, and a consistent way to have data to be predicted brought to the model when new data is being investigated. Another assumption that could be had is that the data being collected can be labeled, to help test and train a model. Otherwise, a deep neural network with high pattern recognition, and high compute cost, would be needed to build the model and change slightly some of the deployment and monitoring aspects brought into the situation. Without that, it is not impossible but does have a higher cost associated with it. 
* **System Requirements:** Some functional requirements that will need to be met are able to detect fraud in a large amount of data, where fraud may not be very common compared to the sheer amount of data. It may also need to detect possible fraud that is on the lower likelihood but still possible. Non-functional would need to improve on the current system that is being used to make more money. 

## RISK & UNCERTAINTIES

* What are potential risks and harms associated with this project?
Potential harms in this would be that fraud is simply to hard for current technology to detect, causing a waste in time. That is unlikely, but what is more likely is the same risks that are currently in use, such as false positive of detecting fraud which would lead to lower customer satisifcation, and false negatives in fraud, not detecting fraud when it does exist, causing a loss in profitability. Another risk that may happen, is that the data being taken in is not labeled, so that means training on data could be difficult as it would be clustering and pattern recognition to find fraud, as there is not always a method of determining if fraud actually happened, so a solution would rely totally on pattern recognition. 
* What are the potential causes of errors or mistakes in the ML system?
Potential errors or mistakes in the ML system could include some outliers in the system that would make a normal transaction appear as fraud. For example, someone truthfully paid and refunded a transaction which is a sign of fraud in normality, but can happen in day to day. Another example would be someone buying something in another location without letting the credit card company know, for example a vacation to another country. Outside transactions without warning are cause for warning. 

# <u>IMPLEMENTATION (How's)</u>

## DEVELOPMENT

* **Methodology:** There are several methodologies that will need to be considered when developing this algorithm. They will be listed below for explaining each part. 
Data Engineering Pipeline: this class shall take in the data, transform it, and then load it into a format that will be used in the algorithm. Otherwise known as the ETL, extract, transform, and load, method. The expectation is that there is a csv called transactions.csv in the directory the code is running that is in a dataframe format with the features given as an example in the docker container. It will extract the data into a pandas dataframe, transform it based on highest likelihood of useful features, then send it back into a file called transactions_transformed.csv to be used to train the model. 
#
Dataset Partitioning: This class will take in a pandas dataframe, like what was given above, and split it in a stratified kfold which can be specified by the value splits=5 in the class. Then it will do a stratified k fold to split the data into folds. Then the get_validation_dataset, get_training_dataset, and get_testing_dataset will all grab the specified dataset based on the fold provided. This means that all of the data will be used, and can be randomized with the variables shuffled=False, and random=None, which are disabled by default. This is the data that the model will take in to be trained.
#
Metrics: the Metrics class is one that has a lot going on for it. The thought process behind the metrics class was that having more metrics that are unused is better then having to little metrics that are not used at all. So many of the typical metrics are used, such as: Precision, Recall, specificity, F1 Score, ROC AUC Score, Accuracy Score. In this model, negative=0(nonfraud) and positive=1(fraud) The uses of each are as follows:

* Precision -- This is the ration of true positives over the total, so TP/TP+FP. This method will be useful in determining how many times we are predicting fraud over how much fraud there actually is. So we want a high precision here, and the previous model was around 40% which means a lot of false positive fraud cases that we will need to track down.
* Recall (sensitivity) -- this is the ratio of tru positives to the total of actual positive observations, so TP/TP+FN which is more useful to us then precision, as it keeps track of calling a ton of non-fraudulent reports fraudulent. A useful metric to keep track of
* Specificity -- This is the ratio of True Negative over true negative plus false positive, so TN/TN+FP. Now this is fairly useful to the model as it will tell how many times it is predicting non-fraudulent transactions as fraud. So performing highly on this ratio is desired. While false positives are not that harmful, keeping them low is beneficial as it means that the model is not missing very many fraudulent transactions while keeping the misses low.
* F1 Score -- this is essentially a score that averages out the recall and precision, which is much stronger for imbalanced dataset and is a weighted average to determine how the model is performing between the two metrics in determining how often, in actuality, is fraud slipping through the model.
* ROC AUC Score -- the full name Reciver Operating Characterisitic area under curve. This score is used to plot the true positive rate against the false positive rate. It will be able to distinguish the models ability to estimate both positive and negative classes to make sure that the model is not just guessing non-fraud and attempting to tell the fraud. This will be useful to determine a balance of how many non-frauds being called fraud vs fraud being called non fraud, and, it is easy to imagine that non-fraud being called fraud is more preferable but still, the model cannot just call everything fraud or that causes more work as well.
* Accuracy Score -- This is the full accuracy of every component, so TP+TN/TP+TN+FP+FN. More useful then precision, but since the dataset is so imbalanced, it will always skew high unless there is something incredibly wrong. Even just guessing non-fraud everytime would give a high accuracy as well, just not as high as precision. 
* The metrics that will be most useful here are specificity to figure out the ratio between how often non-fraud is called fraud and determine how the model can balance that. Next, F1-score and ROC AUC score other both useful for doing the same thing but balancing a weighted average as well as giving a good level to determine where the model is leaning in labeling TP vs TN and seeing how it can be tuned further. Recall (sensitivity) is also useful to determine if the model is correctly identifying positive instances of fraud against how often they are incorrect. 
#
* **High-level System Design:** The system will operate in this fashion -- Transaction -> Bank -> fraud detection model (most likely deep neural network) -> prediction -> fraud or no fraud decision -> let user know if it is fraud. The data shall be processed in an ETL pipeline that will take the transactions we have and limit it to a set of features that are beneficial to the process. So the full process will be to take in the raw data to be read in to a dataframe, process and transform the data, then load it into a new CSV to be used later. The features that will be focused on are listed below:
* Merchant: There looks like a correlation between fraud and the merchant of transaction
* Category: Based on the data there seemed to be clusters of information that correlated around category, amount spent, and when that differed between fraud and non fraud
* Amt: Paired with category, there were distinct differences between fraud and nonfraud
* First: While difficult to tell, there was a correlation between the targets of fraud and the commonality of the name
* Last: same as first
* Sex: This seemed to cluster with names which may be redundant but was not difficult to keep
* Lat: Used for location as it seemed to have some manner or correlation of fraud attackes for location, and is easier to use then city, state, zip
* Long: same as lat
* City_pop: Population had a strange correlation with fraud count, though it could have been just more people more problems
* Job: Strangely, those who had specific job seemed to correlate with category and amount
* Merch_lat: the fraud transaction merchant location did matter when paired with merchent and category
* Merch_long: same as merch_lat
* day_of_Week: When compared, fraud and non fraud had different times of average transaction for day of week, month, and time of day
* day_of_month: same as day_of_week
* time: same as day_of_week
* generation: There was a correlation between likelihood of fraud and the generation someone was born in.
* **Development Workflow:** An overall development process would go like this. Take the current data, and build a model based on the data. Test for recall and precision, iterate on performance with feature engineering or model parameters, test and repeate until performance increases, release, get new data, repeat.

## POLICY

* **Human-Machine Interfaces:** Human interaction will be two fold. First is when the transaction is made and used to predict, and second when a result happens, should it be fraud.
* **Regulations:** Regulations and ethical concerns are something to watch out for. Such as making sure the data being used contains no proprietary user information that should not be shared in any way. Banking information can be critically important to not share, so make sure the data being used is a transaction ID that cannot be tracked, without tying it to a specific account for safety and ethical considerations.

## OPERATIONS

* **Continuous Deployment:** The CD/CI deployment would go like this: Collect  data -> build model -> test -> iterate model parameters and data features -> test -> release if performance increases -> monitor -> repeat
* **Post-deployment Monitoring:** Post deployment, the model and experts should compare on how accurate the model is being, add that to the label of new data, and calculate recall and precision, making sure that the model's performance is not decreasing. O
* **Maintenance:** Outlined slightly in the CD/CI, but take new data that is constantly being collected, have professional label the new data based on results, use the new data for the model to be trained on, train the model. 
* **Quality Assurance:** Quality assurance with fraud detection can be difficult since often there is no way to fully declare fraud vs non-fraud, but since the company has a method of testing the current data and has some values around recall and precision, that is how quality will be tested. Making sure that the numbers for recall and precision are not decreasing, evaluating customer calls about fraud to determine how much profitability and customer satisfaction is being lost due to fraud. The model was designed with modularity in mind in order to make sure that testing and improving is easy. The model is changable simply, the cross_validation is built in in case new testing is needed but not mandatory to save time for sending the model, scaler is separate to try new things. If this model started to drift adjusting and recreating is simple for future proofing.

---