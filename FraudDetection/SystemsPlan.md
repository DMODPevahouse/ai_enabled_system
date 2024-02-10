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

* **Methodology:** Since a solution currently exists, and data is available, an agile methodology is perfect for this use case to make small, incremental changes to a new model to improve on its performance, and integrate it into the bigger system, should performance increase, quickly.
* **High-level System Design:** The system will operate in this fashion -- Transaction -> Bank -> fraud detection model (most likely deep neural network) -> prediction -> fraud or no fraud decision -> let user know if it is fraud.
* **Development Workflow:** An overall development process would go like this. Take the current data, and build a model based on the data. Test for recall and precision, iterate on performance with feature engineering or model parameters, test and repeate until performance increases, release, get new data, repeat.

## POLICY

* **Human-Machine Interfaces:** Human interaction will be two fold. First is when the transaction is made and used to predict, and second when a result happens, should it be fraud.
* **Regulations:** Regulations and ethical concerns are something to watch out for. Such as making sure the data being used contains no proprietary user information that should not be shared in any way. Banking information can be critically important to not share, so make sure the data being used is a transaction ID that cannot be tracked, without tying it to a specific account for safety and ethical considerations.

## OPERATIONS

* **Continuous Deployment:** The CD/CI deployment would go like this: Collect  data -> build model -> test -> iterate model parameters and data features -> test -> release if performance increases -> monitor -> repeat
* **Post-deployment Monitoring:** Post deployment, the model and experts should compare on how accurate the model is being, add that to the label of new data, and calculate recall and precision, making sure that the model's performance is not decreasing. O
* **Maintenance:** Outlined slightly in the CD/CI, but take new data that is constantly being collected, have professional label the new data based on results, use the new data for the model to be trained on, train the model. 
* **Quality Assurance:** Quality assurance with fraud detection can be difficult since often there is no way to fully declare fraud vs non-fraud, but since the company has a method of testing the current data and has some values around recall and precision, that is how quality will be tested. Making sure that the numbers for recall and precision are not decreasing, evaluating customer calls about fraud to determine how much profitability and customer satisfaction is being lost due to fraud. 

---