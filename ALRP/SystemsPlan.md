# <u>MOTIVATION (Why's)</u>

## Why are we solving this problem?

* Problem Statement: The problem the Department of Transportation is facing is that of traffic congestion on the highway. Toll roads are creating a backlock, slowing traffic, and in general hurting the highway driving experience. The solution would be to have a system read the license plate and charge that specific license plate as it drives by, no longer reqiuring anyone to stop and create these problems. 

## Value Proposition

* Why is an ML solution a viable approach? 
Machine learning is an appropriate response to this problem for the fact that when properly trained computer vision models, a system can real-time recognize objects and pull out strings of letters, which is what would need to happen here. The system needs to in real time recognize objects, detect the license plate number, and send out the fee for the toll without falling behind and without being incorrect. 
* Is this problem suitable for machine learning due to its data and desired outcomes?
Since video data is incredibly large, and the outcome is needed to be quick and responsive, this problem is suitable for machine learning, as the data also can be fairly recreated by looking at weather, blurring, etc. So data is highly available in training, and fairly consistent with some outliers of course.
* Can we tolerate potential errors and uncertainties associated with ML?
It depends. The errors should probably skew towards if confidence is not high on the string of the license plate that it should not charge that plate, as charging someone who did not use the toll is unacceptable, but not charging anyone is also not acceptable. Luckily there are steps that can be taken in order to limit those errors, but with proper steps for human in the loop interfaces before bills are sent, errors and uncertainties can be tolerable.
* Is this solution technically feasible with current resources and technologies?
The solution is feasible, but tricky. Video has to be able to provide a model with enough data in order to allow it to pick out the license plate as well as making sure it accounts for errors and anomolies. The model itself can be fairly lightweight once it is trained as well if need be. 

# <u>REQUIREMENTS (What's)</u>

## SCOPE

* What are the overall goals and objectives of this project?
The overall goal of this project is to achieve as close to 0 false license plate numbers as possible, and a high as possible confidence not only on license plates but also on the numbers involved to make sure both those values have a high confidence to use that. 
* How will success be measured for this project?
Success will have to be measured, at first for training by having a human manually determine the string and object annotation of the license plate, which then will use accuracy to determine how well the model takes the string and is completely accurate to every letter in that string. If it is not fully correct, that becomes a problem, so this is actually a model where accuracy does matter, as it has to be completely accurate for a single string to matter, and the model needs to minimize and get as close to 0 as possible.

## REQUIREMENTS

* **Assumptions:** Some key assumptions being attended to is that the data available to be trained on is well labeled and includes a wide variety of weather, license plates types, cars, etc. If a model is to be provided, the assumption would be that it was heavily trained to include those features.
* **System Requirements:** Functional requirements include being able to consistently recieve video in the same format that will allow it to be broken down into a specified amount of images per second to read in the license plates. Other requirements would be that the video would need to be high enough resolution to be read from and have the capability to be manipulated if it helps the data
## RISK & UNCERTAINTIES

* What are potential risks and harms associated with this project?
Potential harms in this would be that people who have not driven on this toll road could get charged if a license plate was drastically misread, or people that have driven on this toll road would not recieve the fee that was needed
* What are the potential causes of errors or mistakes in the ML system?
Causes of errors could be natural distortions, like sun rays, fog, cloudy weather, heavy rain, or human made distortions like tailgating, driving excessively fast that could blur the image, or people clouding up their license plate with stickers, a cloudy gloss, different colors, etc.

# <u>IMPLEMENTATION (How's)</u>

## DEVELOPMENT

* **Methodology:** There are several methodologies that will need to be considered when developing this algorithm. They will be listed below for explaining each part. 
Data Engineering Pipeline: this class shall take in the data, transform it, and then load it into a format that will be used in the algorithm. Otherwise known as the ETL, extract, transform, and load, method. The expectation is that will be a video take in to the extract, the transform will transform the data that would include either bounding the boxes and using that and/or changing the picture with distortion or black/white scale to find the license plate easier will be tested
#
Dataset Partitioning: This class will take in a video format that will be transformed into images and then evaluated on the coco json file to determine annotations and bounding 
# WIP
Metrics: 

* Precision -- This is the ration of true positives over the total, so TP/TP+FP. This method will be useful in determining how many times we are predicting fraud over how much fraud there actually is. So we want a high precision here, and the previous model was around 40% which means a lot of false positive fraud cases that we will need to track down.
* Recall (sensitivity) -- this is the ratio of tru positives to the total of actual positive observations, so TP/TP+FN which is more useful to us then precision, as it keeps track of calling a ton of non-fraudulent reports fraudulent. A useful metric to keep track of
* Specificity -- This is the ratio of True Negative over true negative plus false positive, so TN/TN+FP. Now this is fairly useful to the model as it will tell how many times it is predicting non-fraudulent transactions as fraud. So performing highly on this ratio is desired. While false positives are not that harmful, keeping them low is beneficial as it means that the model is not missing very many fraudulent transactions while keeping the misses low.
* F1 Score -- this is essentially a score that averages out the recall and precision, which is much stronger for imbalanced dataset and is a weighted average to determine how the model is performing between the two metrics in determining how often, in actuality, is fraud slipping through the model.
* ROC AUC Score -- the full name Reciver Operating Characterisitic area under curve. This score is used to plot the true positive rate against the false positive rate. It will be able to distinguish the models ability to estimate both positive and negative classes to make sure that the model is not just guessing non-fraud and attempting to tell the fraud. This will be useful to determine a balance of how many non-frauds being called fraud vs fraud being called non fraud, and, it is easy to imagine that non-fraud being called fraud is more preferable but still, the model cannot just call everything fraud or that causes more work as well.
* Accuracy Score -- This is the full accuracy of every component, so TP+TN/TP+TN+FP+FN. More useful then precision, but since the dataset is so imbalanced, it will always skew high unless there is something incredibly wrong. Even just guessing non-fraud everytime would give a high accuracy as well, just not as high as precision. 
* The metrics that will be most useful here are specificity to figure out the ratio between how often non-fraud is called fraud and determine how the model can balance that. Next, F1-score and ROC AUC score other both useful for doing the same thing but balancing a weighted average as well as giving a good level to determine where the model is leaning in labeling TP vs TN and seeing how it can be tuned further. Recall (sensitivity) is also useful to determine if the model is correctly identifying positive instances of fraud against how often they are incorrect. 
#
* **High-level System Design:** The system will operate in this fashion Video recording from the street -> to the model -> changed from video to images -> transformed, bounded, and possible distorted if it improves performance -> OCR to pull the license plate numbers -> string of the value with lower confince values sent to a human in the loop with an image of the lower confidence -> into the pocket book of those using the toll road. The ETL pipeline shall handle a lot of the transformation needed for the OCR to read in the data.  
# WIP
* 
* **Development Workflow:** An overall development process would go like this. Take the current video, convert it into images, annotate and bound the license plates in the image for testing purposes, have the model bound detect it itself, find the license plate, crop that and have an updated image, return the license plate numbers
## POLICY

* **Human-Machine Interfaces:** Human interaction will be needed for a completely successful model, due to the fact that it will need to make sure that people are not getting charged unfairly, and that the model is recording data correctly. To do so, when the model has a low confidence on a bounding and license plate, that image and license plate value should be sent to a human to evaluate and further determine if hte model needs to be adjusted
* **Regulations:** Regulations and ethical concerns are something to watch out for. The system needs to make sure that license plate data is not being kept for privacy, and even more important is that people who have not used the toll road are not being charged due to a mistake in the system.

## OPERATIONS

* **Continuous Deployment:** The CD/CI deployment would go like this: Collect  video -> change to images -> manipulate and bound images -> ocr -> evaluate string based on labels
* **Post-deployment Monitoring:** Post deployment, the model and experts should compare on how accurate the model is being, add that to making sure the video quality continues to stay the same, as well as that drift does not occur in the model
* **Maintenance:** Outlined slightly in the CD/CI, but accuracy will be used to determine if the model starts to perform less on say a new font used in license plates somewhere, or a new fad style that changes many license plates that may make the model need more training data to be efficient
* **Quality Assurance:** Quality assurance with license plate detection will mainly be determined by pulling out the strings from what the model pulls in from the license plate, looking at the image it got that plate info from, as well as the confidence just to see how the model is determining its confidence level on potentially correct or incorrect values, as well as having a human look over these randomly selected predictions and detections to make sure that the model continues to improve, and this should be happening fairly regularly to make sure that nothing changes drastically.

---
