import yaml
import pdb

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
df = pd.DataFrame

class OutOfRangeException(Exception):
    pass

class Q:
    def __init__(
        self,
        state_set,
        action_set,
        available_actions,
        reward_func,
        transition_func,
        conv=True,
        train_steps=150,
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
        eta=1,
        iota=0.3,
        instant_reward=None,
        display=True,
        maximum_iteration=10000,
        sleep_time=0,
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
        self._custom_params = custom_params if custom_params else {}
        self._custom_show = self._custom_params.get('show', None)
        self._sleep_time = sleep_time
        self._conv = conv
        self._train_steps = train_steps
        self._display = display
        self._maximum_iteration = maximum_iteration
        self._H_table = self.build_q_table() # Heurisitc algorithm
        self._eta = eta
        self._iota = iota
        self._hash_lookup_table = {
            hash(x): x for x in set(self._state_set + self._action_set)
        }

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
        index = [hash(x) for x in self._state_set]
        columns = [hash(x) for x in self._action_set]
        Q_table = df(
            np.zeros(self._dimension),
            index=index,
            columns=columns,
        )
        return Q_table

    @property
    def q_table(self):
        return self._q_table

    def choose_action(self, state):
        available_actions = self._available_actions(state=state)
        if (len(available_actions) == 1):
            return available_actions[0]
        else:
            all_Q = self._q_table.ix[hash(state), :]
            if (np.random.uniform() > self._epsilon) or (all_Q.all() == 0):
                action = random.choice(available_actions)
                #action_ind = np.where(self._action_set == action)[0][0] # Actions must be numpy array
            else:
                action = self._hash_lookup_table.get(all_Q.idxmax())
            return action

    def choose_optimal_action(self, state):
        available_actions = self._available_actions(state=state)
        if (len(available_actions) == 1):
            return available_actions[0]
        else:
            action = self._hash_lookup_table.get(self._q_table.ix[hash(state), :].idxmax())
            return action

    def choose_heuristic_action(self, state):
        available_actions = self._available_actions(state=state)
        if (len(available_actions) == 1):
            return available_actions[0]
        else:
            all_Q = self._q_table.ix[hash(state), :] + self._iota*self._H_table.ix[hash(state), :]
            if (np.random.uniform() > self._epsilon) or (all_Q.all() == 0):
                action = random.choice(available_actions)
                #action_ind = np.where(self._action_set == action)[0][0] # Actions must be numpy array
            else:
                action = self._hash_lookup_table.get(all_Q.idxmax())
            return action

    def train(self, conv=True, heuristic=False):
        '''
        conv: If conv is True, then use the self.convergence as the break condition
        '''
        total_step = self._train_steps
        step = 0
        stop = False
        self._q_table = self.build_q_table()
        self._H_table = self.build_q_table() # Heurisitc algorithm
        init_Q = self._q_table.copy()
        self.conv = []
        while not stop:
            #pdb.set_trace()
            state = self._init_state
            end = False
            if self._display:
                self._display_train_info(train_round=step)
                if self._custom_show:
                    self._custom_show(state=state)
            while not end:
                if not heuristic:
                    action = self.choose_action(state=state)
                else:
                    action = self.choose_heuristic_action(state=state)
                q_predict = self._q_table.ix[hash(state), hash(action)]
                reward = self._reward_func(state=state, action=action)
                next_state = self._transition_func(state=state, action=action)
                if next_state == self._end_state:
                    q = reward
                    end = True
                else:
                    q = reward + self._gamma * self._q_table.ix[hash(next_state), :].max()
                self._H_table.ix[hash(state), :] = 0
                self._H_table.ix[hash(state), hash(action)] = self._q_table.ix[hash(state), :].max() - self._q_table.ix[hash(state), hash(action)] + self._eta
                self._q_table.ix[hash(state), hash(action)] += self._alpha * (q - q_predict)
                state = next_state
                if self._display:
                    if self._custom_show:
                        self._custom_show(state=state)
                        time.sleep(self._sleep_time)
            step += 1
            if self._conv:
                if step >= self._maximum_iteration:
                    raise OutOfRangeException('The iteration time has exceeded the maximum value')
                else:
                    stop = self.convergence(self._q_table.subtract(init_Q))
                    #last_Q = self._q_table.copy()
            else:
                #last_Q = self._q_table.copy()
                self.convergence(self._q_table.subtract(init_Q))
                stop = step == total_step
        #self.plot_conv()
        self._save_q()
        return self.conv

    def plot_conv(self):
        plt.figure()
        plt.plot(range(len(self.conv)), self.conv)

    def run(self):
        if self._run:
            self._run(self.choose_optimal_action)

    def convergence(self, delta_Q=None):
        Q_sum = delta_Q.sum().sum()
        self.conv.append(Q_sum)
        if len(self.conv) > 2 and 0 <= Q_sum - self.conv[-2] < self._phi:
            return True
        else:
            return False


