import yaml
import pdb
import json
import os

import numpy as np
import random
import pandas as pd
import time
import pathlib
import functools

from src.analysis import Result

file_path = pathlib.Path(__file__).parent
defaultQfile = file_path / 'Q.csv'
defaultConvfile = file_path / 'q_sum.csv'

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
        max_train_episodes=1000,
        q_file=defaultQfile,
        load=False,
        epsilon_base=0.1,
        epsilon_decay_rate=1,
        gamma=1,
        learning_rate=0.1,
        phi=1e-4,
        eta=0.9,
        lmd=0.9, # lambda-return
        train_render=False,
        train_render_interval=0,
        train_render_clear=False,
        termination_type="episode",
        termination_precision=None,
        heuristic=False,
        initial_q_mode="zero",
        info_episodes=100,
        result=None,
    ):
        # Define state and action
        self.env = env
        self.ahook = ahook
        self.action_filter = getattr(env, "action_filter", lambda state: self.env.action_space)
        self.result_path = pathlib.Path("results")
        state_len, action_len = len(self.env.observation_space), len(self.env.action_space)
        self.dimension = (state_len, action_len)
        self.q_file = q_file
        self.train_render = train_render
        self.train_render_interval = train_render_interval
        self.train_render_clear = train_render_clear
        self.termination_type = termination_type
        self.termination_precision = termination_precision
        self.max_train_episodes = max_train_episodes
        self.info_episodes = info_episodes
        self.eta = eta
        self.lmd = lmd

        self._q_init_func = {
            "large": lambda dim: 100*np.ones(dim),
            "zero": np.zeros,
            "small": lambda dim: -100*np.ones(dim),
            "random": np.random.random,
        }
        
        # Generate Q table
        if load:
            if not self.q_file.is_file():
                raise QFileNotFoundError('Please check if the Q file exists at: {}'.format(q_file))
            self.q_table = self.load_q(file_name=self.q_file)
        else:
            self.q_table = self.build_q_table(initial_q_mode)
        self.q_table_backup = self.q_table.copy()

        self.epsilon_base = epsilon_base
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.phi = phi

        self.B = 100
        self.A = self.learning_rate * self.B
        self.heuristic = heuristic
        # Eligibility Trace
        self.eligibility_trace = self.build_q_table(mode="zero")
        def _clear(self, target, mode="zero"):
            df = getattr(self, target)
            for col in df.columns:
                df[col].values[:] = 0
        # Set eligitbility trace to zero
        self._clear_et = functools.partial(_clear, self, "eligibility_trace")

        # This is an instance of Result that is used to store all results during training
        self.result = result

        #self.train_dict = {
        #    'Q_learning': self.Q_learning_train,
        #    'SARSA': self.SARSA_train,
        #    'DoubleQ': self.DoubleQ_train,
        #    "SARSA_lambda": self.SARSA_lambda_train,
        #}
        #self.train_algorithm = algorithm
        #if self.train_algorithm == 'DoubleQ':
        #    self.qb_table = self._q_table.copy()
        #elif self.train_algorithm == "SARSA_lambda":
        #    self.eligibility_trace = self.build_q_table(mode="zero")
        #            ##self.train = self.train_dict.get(self.train_algorithm)

    def reset(self):
        self._clear_et()
        self.q_table = self.q_table_backup.copy()

    def load_config(self, config):
        seq = ['epsilon_base', 'gamma', 'alpha', 'phi']
        self.epsilon_base, self.gamma, self.alpha, self.phi = [config.get(x) for x in seq]

    def load_q(self, file_name=None):
        if file_name is None:
            file_name = self.q_file
        self.q_table = pd.read_csv(file_name, header=None, index_col=False)
        self.q_table.columns = self.env.action_space
        self.q_table.index = pd.MultiIndex.from_tuples(self.env.observation_space)
        return self.q_table

    def save_q(self):
        self.q_table.to_csv(self.q_file, index=False, header=False)

    def save_conv(self, filename, conv):
        np.savetxt(filename, conv)

    def save_reward(self):
        return np.array(self.reward_per_episode).mean()

    def render(self):
        if self.train_render:
            if self.train_render_clear:
                os.system("clear")
            time.sleep(self.train_render_interval)
            self.env.render()

    def display_episode_info(self, episode, q_sum, episode_reward, force=False):
        if force or not episode % self.info_episodes:
            if episode > 2:
                diff = q_sum[episode - 1] - q_sum[episode - 2]
            else:
                diff = 0
            print(f"Training Info: episode: {episode}, Q sum: {q_sum[episode - 1]}, ",
            f"episode reward: {episode_reward}, Q Convergence: {diff}")
        else:
            pass
        #print('Convergence: {}'.format(self.conv[-1] - self.conv[-2] if len(self.conv) > 1 else self.conv[-1]))
        #print('Q table: {}'.format(Q_table))

    def build_q_table(self, mode="zero"):
        func = self._q_init_func[mode]
        index = pd.MultiIndex.from_tuples(self.env.observation_space)
        columns = self.env.action_space
        Q_table = df(
            func(self.dimension), # Q value initialization
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
        self.epsilon = self.epsilon_base * self.epsilon_decay_rate/episode

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
            all_Q = Q_table.loc[state, available]
        else:
            all_Q = Q_table.loc[state, :]
        return all_Q.squeeze().idxmax()

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

    def train(self, algorithm="Q_learning"):
        episode = 0
        stop = False
        q_sum = np.zeros((self.max_train_episodes,))
        episode_total_reward = np.zeros((self.max_train_episodes,))
        self.epsilon = self.epsilon_base
        while episode < self.max_train_episodes:
            state = self.env.reset()
            action = self.epsilon_greedy_policy(state=state)
            if algorithm == "SARSA_lambda" or algorithm == "Q_lambda":
                self._clear_et()
            # self.epsilon_decay(episode)
            done = False
            step = 1
            # Total reward of one episode
            episode_reward = 0
            while not done:
                self.render()
                q = self.q_table.at[state, action]
                next_state, reward, done, info = self.env.step(action)
                next_action = self.epsilon_greedy_policy(state=next_state)
                episode_reward += reward
                if done:
                    td_target = reward
                    exploration = True  # Force to set ET to zero
                else:
                    if algorithm == "SARSA" or algorithm == "SARSA_lambda":
                        target_q = self.q_table.at[next_state, next_action]
                    elif algorithm == "Q_learning" or algorithm == "Q_lambda":
                        target_q = self.q_table.loc[next_state, :].max()
                        if algorithm == "Q_lambda":
                            exploration = not target_q == self.q_table.at[next_state, next_action]
                    elif algorithm == "Average_SARSA":
                        target_q = self.q_table.loc[next_state, :].mean()
                    td_target = reward + self.gamma * target_q
                td_error = td_target - q
                if algorithm == "SARSA_lambda" or algorithm == "Q_lambda":
                    self.eligibility_trace.at[state, action] += 1
                    for s in self.env.observation_space:
                        for a in self.env.action_space:
                            self.q_table.at[s, a] += self.learning_rate* td_error * self.eligibility_trace.at[s, a]
                            if algorithm == "SARSA_lambda":
                                self.eligibility_trace.at[s, a] *= self.gamma * self.lmd
                            elif algorithm == "Q_lambda":
                                if exploration:
                                    self.eligibility_trace.at[s, a] = 0
                                else:
                                    self.eligibility_trace.at[s, a] *= self.gamma * self.lmd
                else:
                    self.q_table.at[state, action] += self.learning_rate* td_error
                state = next_state
                action = next_action
                step += 1
            self.save_q()
            q_sum[episode] = self.q_table.sum().sum()
            episode_total_reward[episode] = episode_reward
            self.display_episode_info(episode=episode, q_sum=q_sum, episode_reward=episode_reward)
            episode += 1
            self.epsilon_decay(episode)
            if self.termination_type == "loss" and episode >= 2:
                convergence = abs(q_sum[episode - 1] - q_sum[episode - 2])
                if convergence < self.termination_precision:
                    break
            q_sum_filename = f"{self.env.name}-{algorithm}-train-Q_sum.txt"
            self.save_conv(self.result_path / q_sum_filename, q_sum)
        self.display_episode_info(episode=episode, q_sum=q_sum, episode_reward=episode_reward, force=True)
        return {
            "episode_number": episode,
            "q_sum": q_sum,
            "episode_total_reward": episode_total_reward,
        }

    def run(self):
        state = self.env.reset()
        self.env.render()
        done = False
        total_reward = 0
        while not done:
            action = self.greedy_policy(state)
            next_state, reward, done, info = self.env.step(action)
            self.env.render()
            total_reward += reward
            state = next_state
        print(f"Total reward: {total_reward}")

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
        if self.conv.size > 2 and 0 <= self.conv[-1] - self.conv[-2] < self.phi:
            return True
        else:
            return False


