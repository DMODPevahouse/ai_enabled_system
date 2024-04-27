# System Plan for email Reinforcement Learning

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


