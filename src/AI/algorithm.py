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

class Agent:
    def __init__(
        self,
        env,
        ahook=None,
        episodes=30,
        config_file=None,
        q_file=defaultQfile,
        conv_file=defaultConvfile,
        load=False,
        custom_params=None,
        epsilon_base=0.1,
        gamma=1,
        alpha=0.03,
        phi=1e-3,
        eta=0.9,
        iota=0.9,
        render=True,
        maximum_iteration=200000,
        sleep_time=0,
        heuristic=False,
        quit_mode='c',
        algorithm='Q',
        action_filter=None,
    ):
        # Define state and action
        self.env = env
        self.ahook = ahook
        if action_filter is None:
            self.action_filter = lambda state: self.env.action_space
        else:
            self.action_filter = action_filter
        state_len, action_len = len(self.env.observation_space), len(self.env.action_space)
        self.dimension = (state_len, action_len)
        self.q_file = q_file
        self.conv_file = conv_file
        self.custom_params = custom_params if custom_params else {}
        self.custom_render = self.custom_params.get('render', None)
        self.sleep_time = sleep_time
        self.episodes = episodes
        self.render_flag = render
        self.maximum_iteration = maximum_iteration
        # self.H_table = self.build_q_table() # Heurisitc algorithm
        self.eta = eta
        self.iota = iota
        
        # Generate Q table
        if load:
            if not self.q_file.is_file():
                raise QFileNotFoundError('Please check if the Q file exists at: {}'.format(q_file))
            self.q_table = self.load_q(file_name=self.q_file)
        else:
            self.q_table = self.build_q_table()

        
        # Config all the necessary parameters
        if config_file:
            config = yaml.load(open(config_file))
            self.load_config(config=config)
        else:
            self.epsilon_base = epsilon_base
            self.gamma = gamma
            self.alpha = alpha
            self.phi = phi

        self.B = 100
        self.A = self.alpha * self.B
        self.heuristic = heuristic
        self.quit_mode = quit_mode

        self.train_dict = {
            'Q': self.Q_train,
            'SARSA': self.SARSA_train,
            'DoubleQ': self.DoubleQ_train,
            "SARSA_lambda": self.SARSA_lambda_train,
        }
        self.train_algorithm = algorithm
        if self.train_algorithm == 'DoubleQ':
            self.qb_table = self._q_table.copy()
        elif self.train_algorithm == "SARSA_lambda":
            self.eligibility_trace = self._q_table.copy()
        self.train = self.train_dict.get(self.train_algorithm)

    def load_config(self, config):
        seq = ['epsilon_base', 'gamma', 'alpha', 'phi']
        self.epsilon_base, self.gamma, self.alpha, self.phi = [config.get(x) for x in seq]

    def load_q(self, file_name=None):
        if file_name is None:
            file_name = self.q_file
        self.q_table = pd.read_csv(file_name, header=None, index_col=False)
        self.q_table.columns = self.env.action_space
        self.q_table.index = self.env.observation_space
        return self.q_table

    def save_q(self):
        self.q_table.to_csv(self.q_file, index=False, header=False)

    def save_conv(self):
        np.savetxt(self.conv_file, self.conv)

    def save_reward(self):
        return np.array(self.reward_per_episode).mean()

    def render(self, state=None, sleep=True):
        if self.render_flag:
            if self.custom_render:
                self.custom_render(state=state)
                if sleep:
                    time.sleep(self.sleep_time)

    def display_episode_info(self, episode):
        if len(self.conv) > 2:
            conv = self.conv[-1] - self.conv[-2]
        else:
            conv = 0
        print('episode: {}, Q sum: {}, Q Convergence: {}'.format(episode, self.conv[episode - 1], conv))
        #print('Convergence: {}'.format(self.conv[-1] - self.conv[-2] if len(self.conv) > 1 else self.conv[-1]))
        #print('Q table: {}'.format(Q_table))

    def build_q_table(self):
        index = self.env.observation_space
        columns = self.env.action_space
        Q_table = df(
            np.zeros(self.dimension), # Q value initialization
            index=index,
            columns=columns,
        )
        return Q_table

    def step_ending(self, step=10):
        self.step_end = True
        self.ending_step = step

    def alpha_log(self, episode):
        return np.log(episode + 1)/(episode + 1)

    def alpha_linear(self, episode):
        return self.A/(self.B + episode)

    def epsilon_linear(self, episode):
        return 1/(episode*10 + 1)

    def epsilon_decay(self, episode):
        self.epsilon = self.epsilon_base/episode

    def epsilon_unchanged(self, episode):
        pass

    def epsilon_greedy_policy(self, state):
        actions = self.action_filter(state)
        if (len(actions) == 1):
            return actions[0]
        else:
            if (np.random.uniform() < self.epsilon):
                action = random.choice(actions)
            else:
                action = self.argmax(self.q_table, state, actions)
            return action

    def greedy_policy(self, state):
        return self.argmax(self.q_table, state=state, available=self.action_filter(state))

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
        available_actions = self.available_actions(state)
        if (len(available_actions) == 1):
            return available_actions[0]
        else:
            all_Q = self.q_table.loc[[state], :] + self.iota*self.H_table.loc[[state], :]
            if (np.random.uniform() > self._epsilon) or all_Q.all().all():
                action = random.choice(available_actions)
            else:
                action = all_Q.idxmax()
            return action
    
    def SARSA_train(self):
        episode = 1
        stop = False
        init_Q = self.q_table.copy()
        self.conv = np.array([0])
        self.epsilon = self.epsilon_base
        while episode < self.episodes:
            state = self.env.reset()
            # self.epsilon_decay(episode)
            action = self.epsilon_greedy_policy(state=state)
            done = False
            step = 1
            self.display_episode_info(episode=episode)
            self.env.render()
            while not done:
                q_predict = self.q_table.loc[[state], [action]].values[0][0]
                next_state, reward, done, info = self.env.step(action)
                if done:
                    q = reward
                else:
                    # Here is the critical difference
                    next_action = self.epsilon_greedy_policy(state=next_state)
                    q = reward + self.gamma * self.q_table.loc[[next_state], [next_action]].values[0][0]
                    state = next_state
                    action = next_action
                self.q_table.loc[[state], [action]] += self.alpha * (q - q_predict) 
                self.env.render()
                step += 1
            print(self.q_table)
            pdb.set_trace()
            episode += 1
            self.conv = np.append(self.conv, self.q_table.sum().sum())
            self.save_q()
            self.save_conv()

    def SARSA_lambda_train(self):
        total_episode = self._train_round
        episode = 1
        stop = False
        while not stop:
            self.env.reset()
            state = self.env.observation
            action = self.choose_action(state=state, itertime=episode)
            self._display_train_info(train_round=train_round)
            self._display(state=state)
            done = False
            step = 1
            current_reward = 0
            while not done:
                q_predict = self._q_table.loc[[state], [action]].values[0][0]
                reward, next_state, done, info = self.env.step(action)
                current_reward += reward
                self._eligibility_trace.loc[[state], [action]] += 1
                next_action = self.choose_action(state=next_state, itertime=episode)
                # TD error
                delta = reward + self._gamma * self._q_table.loc[[next_state], [next_action]] - q_predict
                # Iterate all state and action and update the Q value
                #for 


    def Q_train(self):
        total_round = self._train_round
        train_round = 1
        stop = False
        self._H_table = self.build_q_table() # Heurisitc algorithm
        init_Q = self._q_table.copy()
        self.conv = np.array([0])
        self.reward_per_episode = []
        while not stop:
            self.env.reset()
            state = self.env.observation
            self.move_count = self.build_q_table()
            self._display_train_info(train_round=train_round)
            self._display(state=state)
            done = False
            step = 1
            current_reward = 0
            while not done:
                self._display_train_info(train_round=train_round)
                if not self._heuristic:
                    action = self.choose_action(state=state, itertime=train_round)
                else:
                    action = self.choose_heuristic_action(state=state, train_round=train_round)
                q_predict = self._q_table.loc[[state], [action]].values[0][0]
                reward, next_state, done, info = self.env.step(action)
                current_reward += reward
                self.move_count.loc[[state], [action]] += 1
                if done:
                    q = reward
                    end = True
                    self.reward_per_episode.append(current_reward)
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

    def convergence(self):
        if self.conv.size > 2 and 0 <= self.conv[-1] - self.conv[-2] < self._phi:
            return True
        else:
            return False

