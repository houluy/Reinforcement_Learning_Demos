import yaml
import pdb
import json

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import pathlib

file_path = pathlib.Path(__file__).parent
defaultQfile = file_path / 'Q.csv'
defaultConvfile = file_path / 'conv.csv'

df = pd.DataFrame

class OutOfRangeException(Exception):
    pass

class QFileNotFoundError(OSError):
    pass

class Q:
    def __init__(
        self,
        state_set,
        action_set,
        available_actions,
        reward_func,
        transition_func,
        init,
        ahook=None,
        train_round=300,
        run=None,
        start_states=None,
        init_stateset=None,
        end_states=None,
        start_at=[0],
        end_at=[-1],
        config_file=None,
        q_file=defaultQfile,
        conv_file=defaultConvfile,
        load=False,
        custom_params=None,
        epsilon=None,
        gamma=None,
        alpha=None,
        phi=None,
        eta=0.9,
        iota=0.9,
        display=True,
        maximum_iteration=200000,
        sleep_time=0,
        heuristic=False,
        quit_mode='c',
        algorithm='Q',
    ):
        # Define state and action
        self._state_set = state_set
        if start_states is None:
            self._state_start_at = start_at
            self._state_end_at = end_at
            self._init_state = [self._state_set[x] for x in self._state_start_at]
            self._end_state = [self._state_set[x] for x in self._state_end_at]
        else:
            self._init_state = start_states
            self._end_state = end_states
        self.step_end = False
        self._action_set = action_set
        self._available_actions = available_actions # Map between state and available actions
        self._run = run
        self._init = init
        self.ahook = ahook
        state_len, action_len = len(self._state_set), len(self._action_set)
        self._dimension = (state_len, action_len)
        self._q_file = q_file
        self._conv_file = conv_file
        self._custom_params = custom_params if custom_params else {}
        self._custom_show = self._custom_params.get('show', None)
        self._sleep_time = sleep_time
        self._train_round = train_round
        self._display_flag = display
        self._maximum_iteration = maximum_iteration
        self._H_table = self.build_q_table() # Heurisitc algorithm
        self._eta = eta
        self._iota = iota

        # Reward function
        self._reward_func = reward_func

        # State transition function
        self._transition_func = transition_func

        # Generate Q table
        if load:
            if not self._q_file.is_file():
                raise QFileNotFoundError('Please check if the Q file exists at: {}'.format(q_file))
            self._q_table = self._load_q(file_name=self._q_file)
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
            self._phi = phi

        self.B = 100
        self.A = self._alpha * self.B
        self._heuristic = heuristic
        self._quit_mode = quit_mode

        self.train_dict = {
            'Q': self.Q_train,
            'SARSA': self.SARSA_train,
            'DoubleQ': self.DoubleQ_train,
        }
        self.train_algorithm = algorithm
        if self.train_algorithm == 'DoubleQ':
            self._qb_table = self._q_table.copy()
        self.train = self.train_dict.get(self.train_algorithm)

    def Q_wrapper(f):
        def func(self, Q_table=None, *args, **kwargs):
            if Q_table is None:
                Q_table = self._q_table
            return f(self, Q_table=Q_table, *args, **kwargs)
        return func

    def _load_config(self, config):
        seq = ['epsilon', 'gamma', 'alpha', 'phi']
        self._epsilon, self._gamma, self._alpha, self._phi = [config.get(x) for x in seq]

    def _load_q(self, file_name=None):
        if file_name is None:
            file_name = self._q_file
        self._q_table = pd.read_csv(file_name, header=None, index_col=False)
        self._q_table.columns = self._action_set
        self._q_table.index = self._state_set
        return self._q_table

    @Q_wrapper
    def save_q(self, Q_table):
        Q_table.to_csv(self._q_file, index=False, header=False)

    def _save_conv(self):
        np.savetxt(self._conv_file, self.conv)

    def _save_reward(self):
        return np.array(self.reward_per_episode).mean()

    def _display(self, state=None, sleep=True):
        if self._display_flag:
            if self._custom_show:
                self._custom_show(state=state)
                if sleep:
                    time.sleep(self._sleep_time)

    @Q_wrapper
    def _display_train_info(self, train_round, Q_table=None):
        if len(self.conv) > 2:
            conv = self.conv[train_round - 1] - self.conv[train_round - 2]
        else:
            conv = 0
        print('train_round: {}, Convergence: {}'.format(train_round, conv))
        #print('Convergence: {}'.format(self.conv[-1] - self.conv[-2] if len(self.conv) > 1 else self.conv[-1]))
        #print('Q table: {}'.format(Q_table))

    def build_q_table(self):
        index = self._state_set
        columns = self._action_set
        Q_table = df(
            #np.random.rand(*self._dimension)*(-100),
            np.zeros(self._dimension),
            index=index,
            columns=columns,
        )
        return Q_table

    def step_ending(self, step=10):
        self.step_end = True
        self.ending_step = step

    @property
    def q_table(self):
        return self._q_table

    def alpha_log(self, itertime):
        return np.log(itertime + 1)/(itertime + 1)

    def alpha_linear(self, itertime):
        return self.A/(self.B + itertime)

    def epsilon_linear(self, itertime):
        return 1/(itertime*10 + 1)

    @Q_wrapper
    def choose_action(self, state, itertime, Q_table=None):
        available_actions = self._available_actions(state=state)
        if (len(available_actions) == 1):
            return available_actions[0]
        else:
            if (np.random.uniform() < self.epsilon_linear(itertime)):
                action = random.choice(available_actions)
            else:
                action = self.argmax(Q_table, state, available_actions)
            return action

    def choose_optimal_action(self, state):
        return self.argmax(self._q_table, state=state, available=self._available_actions(state=state))

    @staticmethod
    def argmax(Q_table, state, available=None):
        if available:
            all_Q = Q_table.loc[[state], available]
        else:
            all_Q = Q_table.loc[[state], :]
        mval = all_Q.max().max()
        allmaxactions = all_Q[all_Q == mval].dropna(axis=1).columns
        return np.random.choice(allmaxactions)

    def choose_heuristic_action(self, state):
        available_actions = self._available_actions(state=state)
        if (len(available_actions) == 1):
            return available_actions[0]
        else:
            all_Q = self._q_table.loc[[state], :] + self._iota*self._H_table.loc[[state], :]
            if (np.random.uniform() > self._epsilon) or all_Q.all().all():
                action = random.choice(available_actions)
            else:
                action = all_Q.idxmax()
            return action
    
    def SARSA_train(self):
        total_round = self._train_round
        train_round = 1
        stop = False
        init_Q = self._q_table.copy()
        self.conv = np.array([0])
        while not stop:
            self._init()
            state = random.choice(self._init_state)
            #self._init_state = state
            end = False
            step = 1
            self._display()
            while not end:
                #pdb.set_trace()
                self._display_train_info(train_round=train_round)
                action = self.choose_action(state=state, itertime=train_round)
                q_predict = self._q_table.loc[[state], [action]].values[0][0]
                reward = self._reward_func(state=state, action=action)
                next_state = self._transition_func(state=state, action=action)
                if next_state in self._end_state:
                    q = reward
                    end = True
                else:
                    # Here is the critical difference
                    next_action = self.choose_action(state=next_state, itertime=train_round)
                    q = reward + self._gamma * self._q_table.loc[[next_state], [next_action]].values[0][0]
                self._q_table.loc[[state], [action]] += self.alpha_log(train_round) * (q - q_predict) 
                state = next_state
                self._display(state=state)
                step += 1
            train_round += 1
            if self._quit_mode == 'c':
                if step >= self._maximum_iteration:
                    raise OutOfRangeException('The iteration time has exceeded the maximum value')
                else:
                    stop = self.convergence()
            else:
                #self.convergence(self._q_table.subtract(init_Q))
                self.conv = np.append(self.conv, self._q_table.sum().sum())
                stop = (train_round == total_round)

            self.save_q()
            self._save_conv()


    def Q_train(self):
        #pdb.set_trace()
        total_round = self._train_round
        train_round = 1
        stop = False
        self._H_table = self.build_q_table() # Heurisitc algorithm
        init_Q = self._q_table.copy()
        self.conv = np.array([0])
        self.reward_per_episode = []
        #pdb.set_trace()
        while not stop:
            self._init()
            self.move_count = self.build_q_table()
            state = random.choice(self._init_state)
            #self._init_state = state
            self._display_train_info(train_round=train_round)
            self._display(state=state)
            end = False
            step = 1
            current_reward = 0
            while not end:
                #pdb.set_trace()
                self._display_train_info(train_round=train_round)
                if not self._heuristic:
                    action = self.choose_action(state=state, itertime=train_round)
                else:
                    action = self.choose_heuristic_action(state=state, train_round=train_round)
                q_predict = self._q_table.loc[[state], [action]].values[0][0]
                reward = self._reward_func(state=state, action=action)
                current_reward += reward
                next_state = self._transition_func(state=state, action=action)
                self.move_count.loc[[state], [action]] += 1
                if self.step_end:
                    if step == self.ending_step - 1:
                        q = reward
                        end = True
                        self.reward_per_episode.append(current_reward)
                    else:
                        q = reward + self._gamma * self._q_table.loc[[next_state], :].max().max()
                elif (next_state in self._end_state):
                    q = reward
                    end = True
                else:
                    q = reward + self._gamma * self._q_table.loc[[next_state], :].max().max()
                self._H_table.loc[[state], :] = 0
                self._H_table.loc[[state], [action]] = self._q_table.loc[[state], :].max().max() - self._q_table.loc[[state], [action]] + self._eta
                self._q_table.loc[[state], [action]] += self.alpha_log(train_round) * (q - q_predict)
                state = next_state
                self._display(state=state)
                step += 1
            train_round += 1
            if self._quit_mode == 'c':
                if step >= self._maximum_iteration:
                    raise OutOfRangeException('The iteration time has exceeded the maximum value')
                else:
                    stop = self.convergence()
            else:
                #self.convergence(self._q_table.subtract(init_Q))
                self.conv = np.append(self.conv, self._q_table.sum().sum())
                stop = (train_round == total_round)
            self.save_q()
            self._save_conv()
        if self.ahook:
            self.ahook()
        return self.conv

    def DoubleQ_train(self):
        total_round = self._train_round
        train_round = 1
        trainb_round = 1
        astep = 1
        bstep = 1
        stop = False
        self.conv = np.array([0])
        while not stop:
            self._init()
            state = random.choice(self._init_state)
            self._display(state=state)
            end = False
            while not end:
                Q_average = (self._q_table + self._qb_table)/2
                action = self.choose_action(state=state, itertime=train_round, Q_table=Q_average)
                reward = self._reward_func(state=state, action=action)
                next_state = self._transition_func(state=state, action=action)
                # Here is the Double Q process
                e = np.random.rand()
                if e < 0.5:
                    update_Q, estimate_Q = self._q_table, self._qb_table
                    astep += 1
                    com_step = astep
                    #train_round += 1
                    #tr = train_round
                else:
                    update_Q, estimate_Q = self._qb_table, self._q_table
                    bstep += 1
                    com_step = bstep
                    #trainb_round += 1
                    #tr = trainb_round
                q_predict = update_Q.loc[[state], [action]].values[0][0]
                max_action = self.argmax(update_Q, state=next_state)
                if next_state in self._end_state:
                    q = reward
                    end = True
                else:
                    q = reward + self._gamma * estimate_Q.loc[[next_state], [max_action]].values[0][0]
                update_Q.loc[[state], [action]] += 0.1 * (q - q_predict)#self.alpha_log(train_round) * (q - q_predict)
                state = next_state
                self._display(state=state)
            train_round += 1

            Q_average = (self._q_table + self._qb_table)/2
            if self._quit_mode == 'c':
                if train_round >= self._maximum_iteration:
                    raise OutOfRangeException('The iteration time has exceeded the maximum value')
                else:
                    stop = self.convergence()
            else:
                self.conv = np.append(self.conv, Q_average.sum().sum())
                stop = (train_round == total_round)
            self.save_q(Q_table=Q_average)
            self._save_conv()
        return self.conv


    def plot_conv(self):
        plt.figure()
        plt.plot(range(len(self.conv)), self.conv)

    def exlongterm(self):
        return self._q_table.max(axis=1).sum()/self._dimension[0]

    def exlongterm2(self):
        reward = 0
        rsu_set = [[x for x in self._action_set if x[0] == y] for y in range(4)]
        reward2 = 0
        for s in self._state_set:
            ar = 0
            for aset in rsu_set:
                q_max = self._q_table.loc[[s], aset]
                max_val = q_max.max().max()
                max_action = q_max[q_max == max_val].dropna(axis=1).columns
                ar += q_max.max().max()
                rand = np.random.rand()
                max_action = max_action.values[0]
                if rand < 0.7:
                    r = self._reward_func(s, max_action, 1)
                else:
                    r = self._reward_func(s, max_action, 0)
                reward2 += r
            ar /= 4
            reward += ar
        return reward/self._dimension[0], reward2

    def run(self):
        self._run(self.choose_optimal_action)

    def convergence(self):
        if self.conv.size > 2 and 0 <= self.conv[-1] - self.conv[-2] < self._phi:
            return True
        else:
            return False

    def start(self, mode=True):
        if mode:
            return self.train()
        else:
            return Q.run(self)
