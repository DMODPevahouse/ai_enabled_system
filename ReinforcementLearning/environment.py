import gym
import numpy as np


class EmailEnvironment(gym.Env):
    def __init__(self, data):
        self.data = data
        self.current_state_index = 0

    def reset(self):
        self.current_state_index = 0
        return self.data[self.current_state_index][1:]  # Exclude the SubjectLine_ID

    def step(self, action):
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
        return [1, 2, 3]

    def reward(self, action):
        next_state_index = self.current_state_index + 1
        if next_state_index >= len(self.data):
            return 0  # Terminal state

        if self.data[self.current_state_index][-1] == 1:
            return 10
        elif self.data[self.current_state_index][-2] == 1:
            return 5
        else:
            return -10
