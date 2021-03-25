import random
import pdb
import sys
import time
import pandas as pd
import numpy as np
df = pd.DataFrame
import os.path as path
import functools

from src.bases import *

dir_path = path.dirname(path.abspath(__file__))
config = path.join(dir_path, 'treasure.yaml')
Q_file = path.join(dir_path, 'TreasureQ.csv')

from colorline import cprint

oprint = functools.partial(cprint, color='b', bcolor='k', end='')
tprint = functools.partial(cprint, color='r', bcolor='k', end='')

class TreasureHunt:
    def __init__(self, size):
        self.size = size
        self.observation_space = list(range(self.size))
        # Positions
        self.treasure_pos = self.observation_space[-1]
        self.trap_pos = self.observation_space[0]
        self.observation = self.reset() # initial position
        # Rewards
        self.win_reward = 10
        self.lose_reward = -10
        self.wander_reward = -0.05
        self.reward_func = {
            self.treasure_pos: self.win_reward,
            self.trap_pos: self.lose_reward,
        }
        # Symbols
        self.warrior_sign = 'o'
        self.treasure_sign = 'T'
        self.trap_sign = 'X'
        self.path_sign = '_'
        self.left = -1
        self.right = 1
        self.action_space = [self.left, self.right]
        self.custom_params = {
            'show': self.render,
        }

    def render(self, mode="human"):
        if mode == "human":
            pstate = self.observation
            for i in range(self.size):
                if i == pstate:
                    oprint(self.warrior_sign)
                elif i == self.treasure_pos:
                    tprint(self.treasure_sign)
                elif i == self.trap_pos:
                    tprint(self.trap_sign)
                else:
                    print(self.path_sign, end='')
            print()
        else:
            pass

    def step(self, action):
        self.observation = next_state = self.observation + action
        reward = self.reward_func.get(next_state, self.wander_reward)
        if reward == self.wander_reward:
            done = False
        else:
            done = True
        info = {}
        return next_state, reward, done, info

    def reset(self):
        self.observation = self.observation_space[int(self.size/2)]
        return self.observation

    def close(self):
        pass

    def seed(self):
        pass

    def sample_run(self):
        done = False
        self.observation = state = self.reset()
        self.render()
        while not done:
            action = random.choice(self.action_space)
            next_state, reward, done, info = self.step(action)
            self.observation = state = next_state
            self.render()


