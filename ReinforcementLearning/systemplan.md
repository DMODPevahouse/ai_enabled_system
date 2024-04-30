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
Reward here can be classified based on, what the customer would like us to figure out, how to get customers that are
emailed to respond to the email. Respond here is categorized as clicking the email within 24 hours of receiving it.
So that means we have to reward the model based on the number of emails that it gets the user to click on within that 
timeframe. So a new feature was made here to determine the reward, which is the combination of the Sent and recieved.
This feature has been named one_day_response. Since we have some ground truth of the environment, we will be able to
reward the model based on the number of emails that are clicked on within 24 hours of receiving the email based on 
the q-learning table that is created, the models action, and the ground truth of the environment.



Context: 

I have this custom environment
```
import gym

class CustomEnv(gym.Env):
    def __init__(self, data):
        self.data = data
        self.current_state_index = 0

    # Other methods will be defined here
    def reset(self):
        self.current_state_index = 0
        return self.data[self.current_state_index][1:]  # Exclude the SubjectLine_ID
    
    def action_space(self):
        return gym.spaces.Discrete(len(self.data))
    
    def reward(self, action):
        next_state_index = self.current_state_index + 1
        if next_state_index >= len(self.data):
            return 0  # Terminal state

        if self.data[next_state_index][-1] == 1:
            return 1
        else:
            return -1
    
    def step(self, action):
        assert self.action_space().contains(action), f"Action {action} not in the action space"

        prev_state = self.data[self.current_state_index][1:]
        self.current_state_index += 1

        if self.current_state_index >= len(self.data):
            next_state = None
            done = True
            reward = 0
        else:
            next_state = self.data[self.current_state_index][1:]
            reward = self.reward(action)
            done = False

        return next_state, reward, done, {}
```

That is built on this data:

```
		SubjectLine_ID	Gender	Type	Email_Address	Age	Tenure	one_day_response
1	0	2	M	B	Jaj2NuUJneD@gmail.com	44	12	0
2	1	2	M	B	Jaj2NuUJneD@gmail.com	44	12	0
3	2	3	M	C	Qtgy0C@msn.com	33	9	0
4	3	3	M	C	Qtgy0C@msn.com	33	9	0
5	4	2	M	C	JQVjAP2eVCnIz@hotmail.com	26	21	0
6	5	2	M	C	JQVjAP2eVCnIz@hotmail.com	26	21	0
7	6	2	M	C	JQVjAP2eVCnIz@hotmail.com	26	21	1
```

Can you code an example reinforcement learning model on this? 

