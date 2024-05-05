# System Plan for email Reinforcement Learning

# <u>MOTIVATION (Why's)</u>

## Why are we solving this problem?

* Problem Statement: 
    * The problem that we are solving is to determine the best subject line to send to a user based on their 
    information. This is a problem that is faced by many companies that send out emails to their customers. 
    The main goal is to get the user to click on the email, and respond to the email within 24 hours of receiving it. 
    This is a problem that can be solved by reinforcement learning, since the model can take in the data of the userbase
    is being trained on, and take in a set number of subject lines to determine which subject line would be best for
    the user.

## Value Proposition
* **Why is Reinforcement Learning the best approach?**

  Reinforcement learning is a great approach for this problem. Although most people think of RL as being meant in a
game environment, it can be used in a real world scenario. The environment in this case is simply the userbase that is being
used, and the actions are simply the subject lines that are being sent to the user. More detail on this later.
* **Is this problem suitable for Reinforcement Learning due to its data and desired outcome?**

  RL is suitable for this problem since the data can be set up in a way that emulates and environment that remains consistent
with actions that can be deterministic and lead to specific results, based on the data that is given.
* **Can we tolerate potential errors and uncertainty in the model?**
  We can tolerate a certain amount of errors in this model, as if the model is incorrect about what is the best email to
send the user, the email will be simply ignored. So the errors are benign unless the model is so belligerently wrong that
it is recommending emails that are the least likely to be clicked on.
# <u>REQUIREMENTS (What's)</u>

## SCOPE

* **What are the overall goals and objectives of this project?**

  The overall objective this project seeks to accomplish is to determine the best subject line of emails to send to its 
userbase that have the highest likelihood of being responded too within 24 hours as that is most likely to be attached to
a more involved user in whatever the email is about.
* **How will success be measured for this project?**

  Overall success for this project will be measured in the reward achieved by the model on the test emails. The higher
the reward, the more successful the model is expected to be in a real world scenario.
## REQUIREMENTS

* **Assumptions:** 
  
* * **System Requirements:** This system needs to be robust enough to establish a reviews qualifications to a specific star level, however a robust model is not the only requirement. Reviews are sent in by the hundreds per minute, if not more so, which means the system has to keep performance in mind to make sure to not make the application so slow that people would not want to use it at all. A typical review is instant when the information is inserted, if this system is accurate and reliable but takes minutes upon recieving the data making it unseemly to use, that will not fly. 

## RISK & UNCERTAINTIES

* What are potential risks and harms associated with this project?
The potential risks and harm of this project are fairly low. The main negative outcome that is possible would be a mislabeling of a star value. The outcome of this would mean the potential of a bad review to have a positive star rating, or a good review have a negative. Mostly this could affect the trust that people would have in this system to accurately give labels, or concern about abuse of the system. 
* What are the potential causes of errors or mistakes in the ML system?
Errors in this system can be caused by a multitude of reasons. Lack of data, sarcasm in reviews, over critical positive reviews, under critical negative reviews, etc. Avoiding these kinds of results may help, but they could also be common review types that a system would need to be trained on. So with the understand of the modal model, it may need to be trained on exactly this kind of data to try and pick apart the differences in this type of data. 

# <u>IMPLEMENTATION (How's)</u>

## DEVELOPMENT

* **Methodology:** The thought process behind this project is to use the QLearning algorithm to analyze the userbase emails
assisted by the send/responded data to train the model to determine the best subject line to send to the user. This will be done
by reading in the userbase that has determined if a user has clicked on the email at all, and within 24 hours to determine the
reward. The model will then be trained on this data to determine the best subject line to send to the user. The gamma, epsilon,
episodes, alpha, state, action and reward are explained below.
### Gamma
The gamma here that had the best results ended up being 0.99. This is because the model was able to learn the best subject line
to send to the user based on the data that was given. The gamma here is the discount factor that is used to determine the
importance of future rewards. The higher the gamma, the more importance is given to future rewards. 
### Epsilon
Epsilon here was selected to be 0.1. Through testing this was found to be the most successful epsilon value. Epsilon is the
exploration rate, which is the rate at which the model will explore the environment. The higher the epsilon, the more the model
will explore the environment. The lower the epsilon, the more the model will exploit the environment.
### Episodes
From looking at the graph, the models learning levels out at around 35 episodes. So to make sure that the model is trained
properly, the number was set to 50 to insure proper training. The episodes are the number of times the model will run through
the data to train the model.
### Alpha
Testing a few alphas showed that the best alpha was 0.1. Alpha is the learning rate, which is the rate at which the model
will learn from the environment. The higher the alpha, the more the model will learn from the environment. The lower the alpha,
the less the model will learn from the environment. 0.1 seemed to be the best balance for the model to learn the data properly.
### State 
The state that can be used for this problem can be attributed to the description of the details of the email. 
For example, Gender, email address, Type, Age, Tenure, would all be the state of the email. That would mean for the
environment of each learning agent would include those details. Based on the state here, we would have the actions that
can be taken to determine what should be done. Here are reasoning why each was selected, or not selected:
#### Selected
*Type* -- this feature here means if the user is business or consumer, which could greatly affect what action should be taken

*Gender* -- this feature is useful to determine if there is a specific action that affects one gender better than the other

*Age* -- this feature is useful to determine if there is a specific action that affects one age group better than the other

*Tenure* -- this feature is useful to determine if there is a specific action that affects one tenure group better than the other

*Email_address* -- This feature is used to build the state since we are training the data on the userbase that we serve


*Sent and Received* -- these were not directly used, because the combination of the two features creates a new feature that 
is used as the reward. That would make the original features part of the answer that could cause mishap with the model

### Action 
The action that can be taken is listed here as the SubjectLine_ID. This is the ID of a few different subject lines
The main test is to see how certain people click/respond to the email based on the subject line given to them. 
That means, based on the state or environment of the user, should be able for a model to learn and be trained on 
what the best subject line for said user will be. 

### Reward
Reward here can be classified based on, what the customer would like us to figure out, how to get customers that are
emailed to respond to the email. Respond here is categorized as clicking the email within 24 hours of receiving it.
So that means we have to reward the model based on the number of emails that it gets the user to click on within that 
timeframe. So a new feature was made here to determine the reward, which is the combination of the Sent and recieved.
This feature has been named one_day_response. Since we have some ground truth of the environment, we will be able to
reward the model based on the number of emails that are clicked on within 24 hours of receiving the email based on 
the q-learning table that is created, the models action, and the ground truth of the environment.

## Policy
* **Human-Machine Interface:** The human-machine interface is how the model will be used and interacted with. The model will
be used by the company to determine the best subject line to send to the user. A human will use the SubjectLine_ID to determine
the best subject line to send to the user. 
* **Regulations:** There are some concerns that regulations could impact this system. Since the customers data is being
used to train the model, there will need to be some sort of opt-in or opt-out for the customers to determine if 
they are okay with their data being used to train the model. This is a concern that will need to be addressed before
the model can be used in a real world scenario, otherwise there could be legal ramifications.

## Operations
* **Continuous Deployment:** This model can be trained by taking the data, getting new sent/responded emails and using 
new data to validate training of the model. If the stats show that the model is not performing well, then we can retrain
the model with new sent/responded emails to get a better model. This can be done continuously to make sure that the model
is always up to date with the latest data.
* **Post-Deployment Monitoring:** Post deployment monitoring will be done by checking the stats of the model to determine
if the model is performing well. It can also be done to see if the general trend of emails being reponsed to within 24 hours
is increasing. If the trend is increasing, then the model is performing well. If the trend is decreasing, then the model
is not performing well and will need to be retrained with new data.
* **Model Maintenance:** Model maintenance will be done by retraining the model with new data. If the model is not performing
well, use the new data, and train the model on the userbase and the new emails to see what is successful. It would also be
good continue to add new data and possibly new subject lines to improve as well. One caveat to that, is with Q-learning,
model complexity will increase exponentially with the number of states and actions. So it would be good to keep the number
of states and actions to a minimum to keep the model complexity down.
* **Quality Assurance:** Quality assurance will be done by checking the reponsed emails to see if the model is performing well.
* **Distribution:** This model will be trained on a developers work station, then be compiled/pickled and sent to the a
docker container. The docker container will then be sent to a human interacting with the model to determine the best subject
line to send to the user. This process can be automated by another developer to automatically send the subject line to the user
based on the model's recommendation.

## Shortcomings and Improvements
* **Shortcomings:** The model here ended up being very  complicated with the number of states and actions that were used.
This made the model very complex and hard to train. The model also had a hard time learning the data. I had a hard time 
making sure this model could be trained, and in order to do one episode, I let it run for 8 hours and it did not finish
with the full userbase. So unfortunately, I had to cut the userbase down massively to get the model to train. This is a
shortcoming of the model, and would need to be improved upon in the future in order to make use of the whole userbase.
* **Improvements:** A few things I could have done better to improve the model and why it was not done with the time given.
One is using more Numpy and JIT compilers to make the model run faster. I could have also used a more efficient way to
train the model, such as using a neural network to train the model. Once I realized that the complexity of the model was
too high, it was late in the game to optimize qlearning or change to another model and had to run with what I had. I acknowledge
that this was a failure on my part and should have recognized how inefficent the model was going to be, but on small datasets
that I was originally testing on, the model worked exceptionally well. However once I was ready to train the full dataset,
that is when I noticed the models complexity was to much and I could not train the entire dataset. This is a failure on my
part and I should have recognized this earlier and optimized the model to be more efficient.



