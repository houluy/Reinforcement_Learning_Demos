import random
import pdb
import sys
import time
import pandas as pd
import numpy as np
df = pd.DataFrame

import argparse
import yaml
import functools
from Q.Q import Q

from colorline import cprint

oprint = functools.partial(cprint, color='b', bcolor='k', end='')
tprint = functools.partial(cprint, color='r', bcolor='k', end='')

parser = argparse.ArgumentParser(description='This is a demo to show how Q_learning makes agent intelligent')
parser.add_argument('-l', '--load', help='Load Q table from a csv file', action='store_true')
parser.add_argument('-r', '--rounds', help='Training rounds', type=int, default=10)
parser.add_argument('-m', '--mode', help='Mode: oneof ["t"(train), "p"(play)]', choices=['t', 'p'], default='p')
parser.add_argument('-s', '--show', help='Show the training process.', action='store_true', default=False)
parser.add_argument('-c', '--config_file', help='Config file for significant parameters', default=None)

class TreasureHunt:
    def __init__(self, speed=0.5, size=5):
        self._size = size
        self._state_set = list(range(self._size))
        self._action_set = [-1, 1] # 0 stands for left, 1 stands for right
        self._init_state = self._state_set[0]
        self._end_state = self._state_set[-1]
        self._instant_reward = 10
        self._warrior_pos = 0
        self._warrior_sign = 'o'
        self._treasure_pos = self._size - 1
        self._treasure_sign = 'T'
        self._path = '_'
        self._speed = speed # Display speed, lower faster
        self._left = -1
        self._right = 1
        self.custom_params = {
            'show': self.print_map,
        }

    def init(self):
        self.__init__(speed=self._speed, size=self._size)

    @property
    def size(self):
        return self._size

    @property
    def actions(self):
        return self._actions_set

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def warrior_pos(self):
        return self._warrior_pos

    def print_map(self, state=None):
        if not state:
            pstate = self._warrior_pos
        else:
            pstate = state
        for i in range(self._size):
            if i == pstate:
                oprint(self._warrior_sign)
            elif i == self._treasure_pos:
                tprint(self._treasure_sign)
            else:
                print(self._path, end='')
        print()

    def check_win(self, direction, state=None):
        if state is None:
            state = self._warrior_pos
        next_state = self.move(state=state, direction=direction)
        if next_state == self._treasure_pos:
            return True
        else:
            return False

    def _check_move(self, direction, state):
        if direction not in self._action_set:
            raise ValueError('Error direction')
        if state == 0 and direction == 0 or state == self._size and direction == 1:
            raise ValueError('Move out of map')

    def move(self, direction, state=None):
        if state is None:
            cstate = self._warrior_pos
        else:
            cstate = state
        self._check_move(direction=direction, state=cstate)
        cstate += direction
        if state is None:
            self._warrior_pos = cstate
        return cstate

    def run(self):
        end = False
        self.print_map()
        while not end:
            move = np.random.randint(len(self._available_actions) + 1)
            try:
                self.move(move)
            except:
                continue
            self.print_map()
            end = self.check_win()

class Adaptor(TreasureHunt):
    def __init__(self):
        super().__init__()
        self.Q = Q(
            state_set=self._state_set,
            action_set=self._action_set,
            available_actions=self.available_actions,
            reward_func=self.reward,
            transition_func=self.transfer,
            run=self.play,
            instant_reward=self._instant_reward,
            q_file='Q/q.csv',
            config_file='Q/config.yaml',
            custom_params=self.custom_params
        )

    def reward(self, state, action):
        if super().check_win(state=state, direction=action):
            return self._instant_reward
        else:
            return 0

    def transfer(self, state, action):
        next_state = super().move(state=state, direction=action)
        return (next_state, next_state)

    def available_actions(self, state):
        if state == self._state_set[0]:
            return [self.right]
        #elif state == self._state_set[-1]:
        #    return [self.left]
        else:
            return self._action_set

    def train(self):
        self.Q.train()

    def play(self, choose_optimal_action):
        state = self._init_state
        while True:
            action = choose_optimal_action(state=state)
            next_state_ind, next_state = self.transfer(state=state, action=action)
            self.print_map(state=state)
            if super().check_win(state=state, direction=action):
                self.print_map(state=next_state)
                break
            state = next_state

    def run(self):
        self.Q.run()

#class Q:
#    def __init__(self, size=10, epsilon=0.9, speed=0.1, gamma=0.9, alpha=0.1, rounds=15, q_file='q.csv', instant_reward=1, config_file=None, show=False):
#        if config_file:
#            config = yaml.load(open(config_file))
#            self._load_config(config=config)
#        else:
#            self._size = size
#            self._epsilon = epsilon
#            self._gamma = gamma
#            self._alpha = alpha
#            self._instant_reward = instant_reward
#            self._speed = speed
#        self.g = Game(speed=self._speed, size=self._size)
#        self._show = show
#        self._states = list(range(self._size))
#        self._side_state = 0
#        self._current = self.g.warrior_pos
#        self._next = self._current + 1
#        self._actions = self.g.actions
#        self._tran_rounds = rounds
#        self._q_file = q_file
#
#    def _load_q(self):
#        self._q_table = pd.read_csv(self._q_file)
#        self._q_table.columns = self._actions
#        return self._q_table
#
#    def _save_q(self):
#        self._q_table.to_csv(self._q_file, index=False)
#
#    def _load_config(self, config):
#        seq = ['size', 'epsilon', 'gamma', 'alpha', 'instant_reward', 'speed']
#        self._size, self._epsilon, self._gamma, self._alpha, self._instant_reward, self._speed = [config.get(x) for x in seq]
#
#    def build_q_table(self):
#        self._q_table = df(
#            np.zeros((self._size, len(self._actions))),
#            columns=self._actions,
#        )
#        return self._q_table
#
#    def choose_action(self, state):
#        all_actions = self._q_table.iloc[state, :]
#        if (state == self._side_state):
#            action = self.g.right
#        elif (np.random.uniform() > self._epsilon) or (all_actions.all() == 0):
#            action = np.random.choice(self._actions)
#        else:
#            action = all_actions.idxmax()
#        return action
#
#    def choose_action_by_q(self, state, q_table=None):
#        if q_table is None:
#            q_table = self._q_table
#        if (state == self._side_state):
#            return self.g.right
#        return q_table.iloc[state, :].idxmax()
#
#    def move(self, action, state=None):
#        self.move(direction=action)
#        self._next = self.g.warrior_pos
#        return self._next
#
#    def reward(self, state, action):
#        if self.g.check_win(state=state, direction=action):
#            return self._instant_reward
#        else:
#            return 0
#
#    def train(self, load=True):
#        if load:
#            self._load_q()
#        else:
#            self.build_q_table()
#        step = self._tran_rounds
#        while self._tran_rounds > 0:
#            state = 0
#            end = False
#            self.g.init()
#            if self._show:
#                print('Q table:')
#                print(self._q_table)
#                print()
#                print('Training round: %d' % (step - self._tran_rounds + 1))
#                print()
#                self.g.print_map()
#            while not end:
#                action = self.choose_action(state)
#                q_predict = self._q_table.ix[state, action]
#                instant_reward = self.reward(state=state, action=action)
#                next_state = self.move(action=action)
#                if instant_reward:
#                    q = instant_reward
#                    end = True
#                else:
#                    q = instant_reward + self._gamma * self._q_table.iloc[next_state, :].max()
#                self._q_table.ix[state, action] += self._alpha * (q - q_predict)
#                if self._show:
#                    self.g.print_map()
#                state = next_state
#            self._tran_rounds -= 1
#        self._save_q()
#
#    def run(self):
#        self.g.init()
#        self._load_q()
#        state = self.g.warrior_pos
#        self.g.print_map()
#        end = False
#        while not end:
#            action = self.choose_action_by_q(state=state)
#            if self.g.check_win(direction=action):
#                end = True
#            state = self.move(action=action)
#            self.g.print_map()

if __name__ == '__main__':
    agent = Adaptor()
    agent.train()
