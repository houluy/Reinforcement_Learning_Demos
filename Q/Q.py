import yaml

import numpy as py
import pandas as pd
df = pd.DataFrame

class Q:
    def __init__(self, state_set, action_set, reward_func, transit_func, state_init_ind=0, config_file=None, q_file=None, action_set=None, epsilon=None, gamma=None, alpha=None, instant_reward=None):
        # Define state and action
        self._state_set = state_set
        self._state_init_ind = state_init_ind
        self._init_state = self._state_set[self._state_init_ind]
        self._action_set = action_set
        state_len, action_len = len(self._state_set), len(self._action_set)
        self._dimension = (state_len, action_len)

        # Reward function
        self._reward_func = reward_func

        # State transition function
        self._transition = transition_func

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

    def choose_action(self, state_ind):
        all_Q = self._q_table.iloc[state_ind, :]
        if (np.random.uniform() > self._epsilon) or (all_Q.all() == 0):
            action = np.random.choice(self._actions)
            action_ind = np.where(self._actions == action)[0][0] # Actions must be numpy array
        else:
            action = all_Q.max()
            action_ind = all_Q.values.argmax()
        return (action_ind, action)

    def train(self, steps=20):
        step = steps
        while step > 0:
            state = self._init_state
            state_ind = self._init_state_ind
            end = False
            while not end:
                action_ind, action = self.choose_action(state_ind)
                q_predict = self._q_table.ix[state_ind, action_ind]
                instant_reward = self._reward_func(state=state, action=action)
                next_state_ind, next_state = self._transit(state=state, action=action)
                if next_state == self._end_state:
                    q = instant_reward
                    end = True
                else:
                    q = instant_reward + self._gamma * self._q_table.iloc[next_state_ind, :].max()
                self._q_table.ix[state_ind, action_ind] += self._alpha * (q - q_predict)
                state = next_state
                state_ind = next_state_ind
            step -= 1
        self._save_q()

