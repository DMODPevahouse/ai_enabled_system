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

  Reinforcement learning is a great approach for this problem. Althought most people think of RL as being meant in a
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

### State 
The state that can be used for this problem can be attributed to the description of the details of the email. 
For example, Gender, Type, Age, Tenure, would all be the state of the email. That would mean for the
environment of each learning agent would include those details. Based on the state here, we would have the actions that
can be taken to determine what should be done. Here are reasonings why each was selected, or not selected:
#### Selected
*Type* -- this feature here means if the user is business or consumer, which could greatly affect what action should be taken

*Gender* -- this feature is useful to determine if there is a specific action that affects one gender better than the other

*Age* -- this feature is useful to determine if there is a specific action that affects one age group better than the other

*Tenure* -- this feature is useful to determine if there is a specific action that affects one tenure group better than the other
#### Not Selected
*Email_address* -- This feature would only assist in overfitting the data, as in a real world scenario, the email address 
could be completely random, or completely new information that the model would not be trained on

*Customer_ID* -- This feature would only assist in overfitting the data, as in a real world scenario, the customer ID
is just a numbering mechanism that does not describe the user

*Sent and Recieved* -- these were not directly used, because the combination of the two features creates a new feature that 
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




