import yaml

import numpy as py
import pandas as pd
df = pd.DataFrame

class Q:
    def __init__(self, state_set, action_set, reward_func, config_file=None, q_file=None, action_set=None, epsilon=None, gamma=None, alpha=None, instant_reward=None):
        # Define state and action
        self._state_set = state_set
        self._action_set = action_set
        state_len, action_len = len(self._state_set), len(self._action_set)
        self._dimension = (state_len, action_len)

        # Reward function
        self._reward_func = reward_func

        # Generate Q table
        if q_file:
            self._q_table = self._load_q(file_name=q_file)
        else:
            self._q_table = self.build_q_table()

        # Config all the necessary parameters
        if config_file:
            config = yaml.load(open(config_file))
            self._load_config(config=config)
        else:
            self._epsilon = epsilon
            self._gamma = gamma
            self._alpha = alpha
            self._instant_reward = instant_reward

    def _load_config(self, config):
        seq = ['epsilon', 'gamma', 'alpha', 'instant_reward']
        self._epsilon, self._gamma, self._alpha, self._instant_reward = [config.get(x) for x in seq]

    def _load_q(self, file_name=None):
        if file_name is None:
            file_name = self._q_file
        self._q_table = pd.read_csv(file_name)
        return self._q_table

    def _save_q(self):
        self._q_table.to_csv(self._q_file, index=False)

    def build_q_table(self):
        self._q_table = df(
            np.zeros(self._dimension),
            columns=self._actions,
        )
        return self._q_table

    def choose_action(self, state):
        all_actions = self._q_table.iloc[state, :]
        if (np.random.uniform() > self._epsilon) or (all_actions.all() == 0):
            action = np.random.choice(self._actions)
        else:
            action = all_actions.idxmax()
        return action

