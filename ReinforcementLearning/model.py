import numpy as np
import random


class QLearningAgent:
    def __init__(self, env, alpha=0.5, epsilon=0.5, gamma=0.5):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = {}

    def get_q_value(self, state, action):
        state_action_key = (tuple(state), action)
        if state_action_key not in self.q_table:
            return -1.0
        return self.q_table[state_action_key]

    def set_q_value(self, state, action, value):
        state_action_key = (tuple(state), action)
        self.q_table[state_action_key] = value

    def choose_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            # Explore by choosing a random action
            available_actions = self.env.action_space()
            return random.choice(available_actions)
        else:
            # Exploit by choosing the action with the highest Q-value
            q_values = [self.get_q_value(state, a) for a in self.env.action_space()]
            max_q_value = max(q_values)
            available_actions = [a for a in self.env.action_space() if self.get_q_value(state, a) == max_q_value]
            return random.choice(available_actions)

    def learn(self, num_episodes=1):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                q_value = self.get_q_value(state, action)
                next_q_value = 0.0 if done else max(
                    [self.get_q_value(next_state, a) for a in range(len(self.env.data))])
                self.set_q_value(state, action, q_value + self.alpha * (reward + self.gamma * next_q_value - q_value))
                state = next_state

    def play(self, num_episodes=10):
        """
        Plays the agent for a specified number of episodes and returns statistics about its performance.

        Args:
            num_episodes (int): The number of episodes to play.

        Returns:
            dict: A dictionary containing statistics about the agent's performance.
        """
        statistics = {
            "total_reward": 0.0,
            "average_reward": 0.0,
            "max_reward": float('-inf'),
            "min_reward": float('inf'),
            "num_episodes": num_episodes
        }

        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            statistics["total_reward"] += episode_reward
            statistics["average_reward"] = statistics["total_reward"] / (i + 1)
            if episode_reward > statistics["max_reward"]:
                statistics["max_reward"] = episode_reward
            if episode_reward < statistics["min_reward"]:
                statistics["min_reward"] = episode_reward

        return statistics
