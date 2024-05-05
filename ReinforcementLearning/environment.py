import gym
import numpy as np


class EmailEnvironment(gym.Env):
    """
    Custom environment for the email dataset

    The action space is discrete and consists of 3 actions called SubjectLine_ID:

    Args:
        data (np.array): The email dataset

    Attributes:
        data (np.array): The email dataset
        response (np.array): The response of the email
        actions (np.array): The actions of the email
        current_state_index (int): The current state index t
    """
    def __init__(self, data):
        self.data = data[:, :-4]
        self.response = data[:, -3:]
        self.actions = data[:, -4]
        self.current_state_index = 0

    def reset(self):
        """
        Reset the environment to the initial state

        Returns:
            np.array: The initial state
        """
        self.current_state_index = 0
        return self.data[self.current_state_index][1:]  # Exclude the SubjectLine_ID

    def step(self, action):
        """
        Take an action in the environment

        Args:
            action (int): The action to take

        Returns:
            np.array: The next state
            float: The reward
            bool: Whether the episode is done
            dict: Additional information
        """
        assert self.action_space(), f"Action {action} not in the action space"

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

    def action_space(self):
        """
        Get the action space

        Returns:
            list: The action space
        """
        return [1, 2, 3]

    def reward(self, action):
        """
        Get the reward for the action

        Args:
            action (int): The action

        Returns:
            float: The reward
        """
        next_state_index = self.current_state_index + 1
        if next_state_index >= len(self.data):
            return 0  # Terminal state

        if self.response[self.current_state_index][0] == 1:
            if self.actions[self.current_state_index] == action:
                return 10
            else:
                return -10
        elif self.response[self.current_state_index][1] == 1:
            if self.actions[self.current_state_index] == action:
                return 5
            else:
                return -10
        else:
            return -10
