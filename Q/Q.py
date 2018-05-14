import yaml
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
df = pd.DataFrame

class Q:
    def __init__(
        self,
        state_set,
        action_set,
        available_actions,
        reward_func,
        transition_func,
        train_steps=30,
        run=None,
        state_init_ind=0,
        state_end_ind=-1,
        config_file=None,
        q_file=None,
        load_q=True,
        custom_params=None,
        epsilon=None,
        gamma=None,
        alpha=None,
        instant_reward=None,
        sleep_time=0.01,
    ):
        # Define state and action
        self._state_set = state_set
        self._state_init_ind = state_init_ind
        self._state_end_ind = state_end_ind
        self._init_state = self._state_set[self._state_init_ind]
        self._end_state = self._state_set[self._state_end_ind]
        self._action_set = action_set
        self._available_actions = available_actions # Map between state and available actions
        self._run = run
        state_len, action_len = len(self._state_set), len(self._action_set)
        self._dimension = (state_len, action_len)
        self._q_file = q_file
        self._custom_params = custom_params
        self._custom_show = self._custom_params.get('show', None)
        self._sleep_time = sleep_time
        self._train_steps = train_steps
        self.conv = []

        # Reward function
        self._reward_func = reward_func

        # State transition function
        self._transition_func = transition_func

        # Generate Q table
        if load_q:
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
        seq = ['epsilon', 'gamma', 'alpha', 'instant_reward', 'phi']
        self._epsilon, self._gamma, self._alpha, self._instant_reward, self._phi = [config.get(x) for x in seq]

    def _load_q(self, file_name=None):
        if file_name is None:
            file_name = self._q_file
        self._q_table = pd.read_csv(file_name)
        return self._q_table

    def _save_q(self):
        self._q_table.to_csv(self._q_file, index=False)

    def _display_train_info(self, train_round):
        print('train_round: {}'.format(train_round))
        print('Q table: {}'.format(self._q_table))

    def build_q_table(self):
        self._q_table = df(
            np.zeros(self._dimension),
            columns=self._action_set,
        )
        return self._q_table

    def choose_action(self, state):
        available_actions = self._available_actions(state=state)
        if (len(available_actions) == 1):
            return available_actions[0]
        else:
            all_Q = self._q_table.iloc[state, :]
            if (np.random.uniform() > self._epsilon) or (all_Q.all() == 0):
                action = np.random.choice(available_actions)
                #action_ind = np.where(self._action_set == action)[0][0] # Actions must be numpy array
            else:
                action = available_actions[all_Q.values.argmax()]
                #action_ind = all_Q.values.argmax()
            return action

    def choose_optimal_action(self, state):
        available_actions = self._available_actions(state=state)
        if (len(available_actions) == 1):
            return available_actions[0]
        else:
            action = available_actions[self._q_table.iloc[state, :].values.argmax()]
            return action

    def train(self, conv=True):
        '''
        conv: If conv is True, then use the self.convergence as the break condition
        '''
        total_step = self._train_steps
        step = 0
        stop = False
        last_Q = self._q_table.copy()
        while not stop:
            state = self._init_state
            state_ind = self._state_init_ind
            end = False
            self._display_train_info(train_round=step)
            if self._custom_show:
                self._custom_show(state=state)
            while not end:
                # pdb.set_trace()
                action = self.choose_action(state=state)
                q_predict = self._q_table.ix[state, action]
                reward = self._reward_func(state=state, action=action)
                next_state_ind, next_state = self._transition_func(state=state, action=action)
                if next_state == self._end_state:
                    q = reward
                    end = True
                else:
                    q = reward + self._gamma * self._q_table.iloc[next_state_ind, :].max()
                self._q_table.ix[state, action] += self._alpha * (q - q_predict)
                state = next_state
                state_ind = next_state_ind
                if self._custom_show:
                    self._custom_show(state=state)
                    time.sleep(self._sleep_time)
            step += 1
            if conv:
                if step >= 100:
                    stop = True
                stop = self.convergence(self._q_table.subtract(last_Q))
            else:
                last_Q = self._q_table.copy()
                stop = step == total_step
        self._save_q()

    def plot_conv(self):
        plt.plot(range(len(self.conv)), self.conv)

    def run(self):
        if self._run:
            self._run(self.choose_optimal_action)

    def convergence(self, delta_Q=None):
        count = 0
        steps = delta_Q.shape[0]
        for i in range(steps):
            Q_s = delta_Q.ix[i, :]
            Psi = (max(Q_s) - min(Q_s))/2
            count += Psi
        mean = count/steps
        self.conv.append(mean)
        if mean < self._phi:
            return True
        else:
            return False

