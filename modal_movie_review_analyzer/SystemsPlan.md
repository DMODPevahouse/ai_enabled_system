# <u>MOTIVATION (Why's)</u>

## Why are we solving this problem?

* Problem Statement: This problem is defined by the traditional 5 star rating system does not apply enough value to what 5 stars really means. Some may think that a 5 star review has a lot of flaws but leaves it at 5 stars, others may give a horrible review by words and leave a sarcastic 5 star, and this can happen for 0 or 1 stars as well. This problem should be addressed by having others leave a review and the system determine the rating itself instead of the user

## Value Proposition

* Why is an ML solution a viable approach? 
Machine learning is an appropriate response to this problem for the fact that training a model on descriptions and words that have a specific classification is not only feasible but common. A model able to read in the user data and determine what the rating should be based on the review would allow a user to give realistic reviews and the system itself to place the star level to the movie
* Is this problem suitable for machine learning due to its data and desired outcomes?
The problem is definitely suitable for machine learning since we are simply using a classification of star level based on a review. 
* Can we tolerate potential errors and uncertainties associated with ML?
Ultimately, here, errors are mostly benign. If a system gives a 4 star review instead of a 5 star, then unless there are direct punishments for having a low review, that typically do not exist, then its only a slight lowering of star level based on all of the reviews coming in. If there was a direct issue with the system given dramatic and many great errors, that would be another issue itself. But even that being the case there is no real risk of business, safety, or reputation at stake in this system. 
* Is this solution technically feasible with current resources and technologies?
The solution is feasible with current libraries like pytorch or scikit-learn. Many libraries are able to take in text modal data and use that data with an encoding/decoding process to train a model with. Along with that, a system that simple takes a users text and converts it into data to place a category is normal practice

# <u>REQUIREMENTS (What's)</u>

## SCOPE

* What are the overall goals and objectives of this project?
The goals of this project would be to have a system that takes in modal text data and applies a classification of which star level that review would warrant. This will be achieved by taking in the data from preview movie reviews and training a modeol to make classifications based on that data. 
* How will success be measured for this project?
Success will be determined by taking our ground truth data we trained on, and seeing how similar the models classification on that data is. This can be achieved with a k fold cross validation of the data with the main measurements being Mean Average Precision and mean reciprocal rank

## REQUIREMENTS

* **Assumptions:** Assumptions here is that we have data that can train the model on, and along with that there is enough data that is serious enough to be used to determine what a star ranking of said movie would be. If we do not have enough data to establish enough of a baseline of positive vs negative remarks then the model will fail to establish what a 5 star is vs a 0 star review. This would not be acceptable. We also need to assume that there is a way and transfering the reviews to the system, most likely some sort of json file being transfered to the docker container to be read in as a dataframe and converted to how the model was trained. Another key assumption is that there are not a ton of sarcastic/witty reviews that are hard for a system like this to determine if they are serious bad, or serious good reviews. If the data has a large amount of conflicting data like that, it may be difficult for a system to determine where that kind of data should go. 
* **System Requirements:** This system needs to be robust enough to establish a reviews qualifications to a specific star level, however a robust model is not the only requirement. Reviews are sent in by the hundreds per minute, if not more so, which means the system has to keep performance in mind to make sure to not make the application so slow that people would not want to use it at all. A typical review is instant when the information is inserted, if this system is accurate and reliable but takes minutes upon recieving the data making it unseemly to use, that will not fly. 

## RISK & UNCERTAINTIES

* What are potential risks and harms associated with this project?
The potential risks and harm of this project are fairly low. The main negative outcome that is possible would be a mislabeling of a star value. The outcome of this would mean the potential of a bad review to have a positive star rating, or a good review have a negative. Mostly this could affect the trust that people would have in this system to accurately give labels, or concern about abuse of the system. 
* What are the potential causes of errors or mistakes in the ML system?
Errors in this system can be caused by a multitude of reasons. Lack of data, sarcasm in reviews, over critical positive reviews, under critical negative reviews, etc. Avoiding these kinds of results may help, but they could also be common review types that a system would need to be trained on. So with the understand of the modal model, it may need to be trained on exactly this kind of data to try and pick apart the differences in this type of data. 

# <u>IMPLEMENTATION (How's)</u>

## DEVELOPMENT

* **Methodology:** There are several methodologies that will need to be considered when developing this algorithm. They will be listed below for explaining each part. 
Data Engineering Pipeline: this class shall take in the data, transform it, and then load it into a format that will be used in the algorithm. Otherwise known as the ETL, extract, transform, and load, method. The expectation is that there is a csv called amazon_movie_reviews.csv that will be transformed into a useable form and then exported into a cleaned data set that will not include data that would bog the system down and cause issues with performance or labeling. 
#
Dataset Partitioning: This class will take in a pandas dataframe, like what was given above, and split it in a stratified kfold which can be specified by the value splits=5 in the class. Then it will do a stratified k fold to split the data into folds. Then the get_validation_dataset, get_training_dataset, and get_testing_dataset will all grab the specified dataset based on the fold provided. This means that all of the data will be used, and can be randomized with the variables shuffled=False, and random=None, which are disabled by default. This is the data that the model will take in to be trained.
#
Metrics: the main metrics to be used in a model such as this which could place multiple labels on the prediction is mean average precision and mean reciprocal rank. The reason for this is there are typically multiple guesses to be found, with the highest likelihood guess typically selected, but second place results are not worthless as this could mean the model is being successful in applying classifications even if it is not the highest likelihood answer for the model.

* Mean Average Precision -- This is the average precision of what the model would produce, as in if it does produce an answer, or answers, how precise is it to what the data actually shows over all estimations in the calculation
* Mean Reciprocal Rank -- the metric here measures based off of the rank of the value in regards to how it was intended to rank and takes into account both the amount of predicted values and how close they are to the answer

#
* **High-level System Design:** The system will operate in this fashion -- review -> model -> classification of the star level. The system will be trained on in a matter similar to this, csv of data -> transform the data with an ETL pipeline -> train the model -> validate and test the model -> Confirm performance, if not start back at the beginning -> accept new data and transform it to be predicted. Here is the initial testing of the features that will be useful to be tested. 
* Rating: Ground truth to test for performance
* Review Title: title of the movie to be reviewed
* Text: the actual review of the move in text form 
* helpful_vote: how many upvotes this comment had to determine how useful the review was to see how people respond to it
* main_category: category of the movie, used to see if there is a generic population handling of reviews in that feild
* average rating: the overall rating of the movie, could also be used as the ground truth
* rating number: how many reviews of said movie, as this could have a big impact on the type of reviews or the average rating
* **Development Workflow:** An overall development process would go like this. Take the current data, and build a model based on the data. Test for recall and precision, iterate on performance with feature engineering or model parameters, test and repeate until performance increases, release, get new data, repeat.

* **Model selection:** This system was difficult to select a model with, but after several tests with a couple of different models, Logistic Regression was selected. There are several reasons for this model.
* 1. This model allows for ease of taking the top k probability selections to see how the model is placing the data
* 2. Compared to other models, due to the sheer size of this data, it trained relatively fast with respectable performance.
* 3. The flexibility of the model with the data provided. 

* **Preprocessesing and Encoding:** For this, I decided to go with a custom preprocessing method that allowed the removal of a lot of words, punctuation, and tokens that only added complexity without giving more data. Words like the, and, is, punctuation, https, etc. This allowed a smaller dataset to be created but was still quite substantially large. For the encoding method, due to the large dataset, I went with TFIDF, which allowed me to express a lot of specific variables. This dataset had somehwere around 1 million unique words and tokens, which obviously was not going to create avery useful dataset. With TFIDF I was able to lower that down by setting a make feature count, as well as enabling a minimum amount of repition for a word to show up. Since it would be unlikely that any data would be used outside of those parameters. 

* **Data Selection:** As mentioned before, the data was cut significantly already, but the only way to make sure that the model remained small enough for a container was to cut the data significantly down. performance from the modal still held up respectably, but more investigation would be needed.

## POLICY

* **Human-Machine Interfaces:** Here the interaction is obviously that the end goal is for humans to interact with labels to determine the validity of the review and how it will impact their next movies. As for interacting with the model, its very possible to incoporate a system of interfacing with the model to insure that the review classifications are not getting out of hand clearly in the wrong direction of validity
* **Regulations:** Regulations would not be as much of a concern for this system, besides ensuring that there was no unethical tampering of the star data. There would need to be systems in place to avoid over saturation of a movie with positive reviews by abuse of the system to promote fairness, as well as making sure not to accept bribes to hike up the reviews.

## OPERATIONS

* **Continuous Deployment:** The CD/CI deployment would go like this: Collect  data -> build model -> test -> iterate model parameters and data features -> test -> release if performance increases -> monitor -> repeat
* **Post-deployment Monitoring:** Post deployment, the model and experts should compare on how accurate the model is being, add that to the label of new data, and calculate recall and precision, making sure that the model's performance is not decreasing. O
* **Maintenance:** Outlined slightly in the CD/CI, but take new data that is constantly being collected, have professional label the new data based on results, use the new data for the model to be trained on, train the model. This model will be provided by pickle, since the data required to train the model is so huge. So whenever this model needs to be rebuilt or updated, it will need to be re-pickled(?) as well.
* **Quality Assurance:** Quality assurance here is mostly determined on making sure that there is no skew of the model in how it is giving review labels. This could include watching the culture of movie reviews as potentially one genre or a new fad in the public changes how review are done. An example would be people dramatically tearing apart their favorite movie in all its peices but wanting to give a positive review, or very short seemingly derrogative reviews that are meant to be positive, and of course, vise versa.
* **Distribution:** This will be distributed on a docker container that will accept a json file in order to make an accurate rating. It will then return a statement to give the rating. 

---