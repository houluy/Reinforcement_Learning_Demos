import random
import pdb
import sys
import time
import pandas as pd
import numpy as np
df = pd.DataFrame
import os.path as path

import functools
from Q.Q.Q import Q

dir_path = path.dirname(path.abspath(__file__))
config = path.join(dir_path, 'treasure.yaml')
Q_file = path.join(dir_path, 'TreasureQ.csv')

from colorline import cprint

oprint = functools.partial(cprint, color='b', bcolor='k', end='')
tprint = functools.partial(cprint, color='r', bcolor='k', end='')

#oprint = functools.partial(print, end='')
#tprint = functools.partial(print, end='')

class TreasureHunt:
    def __init__(self, speed, size):
        self._size = size
        self._state_set = list(range(self._size))
        self._action_set = [-1, 1] # 0 stands for left, 1 stands for right
        self._init_state_ind = 1#random.randint(1, self._size - 1)
        self._init_state = self._state_set[self._init_state_ind]
        self._end_state_ind = [0, -1]
        self._win_state = self._state_set[-1]
        self._lose_state = self._state_set[0]
        self._win_reward = 1
        self._lose_reward = -1
        self._warrior_pos = self._init_state
        self._warrior_sign = 'o'
        self._treasure_pos = self._size - 1
        self._treasure_sign = 'T'
        self._path_cost = -0.05
        self._trap_pos = 0
        self._trap_sign = 'X'
        self._path = '_'
        self._speed = speed # Display speed, lower faster
        self._left = -1
        self._right = 1
        self._actions = [self._left, self._right]
        self.custom_params = {
            'show': self.print_map,
        }
        self._end_dict = {
            self._win_state: self._win_reward,
            self._lose_state: self._lose_reward,
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
            elif i == self._trap_pos:
                tprint(self._trap_sign)
            else:
                print(self._path, end='')
        print()

    def check_win(self, direction, state=None):
        if state is None:
            state = self._warrior_pos
        next_state = self.move(state=state, direction=direction)
        rew = self._end_dict.get(next_state, None)
        if rew is not None:
            return rew
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
            move = random.choice(self._actions)
            end = self.check_win(direction=move)
            try:
                self.move(move)
            except:
                continue
            self.print_map()

class Adaptor(TreasureHunt, Q):
    def __init__(self, size=10, speed=20, args=None):
        TreasureHunt.__init__(self, speed, size)
        if args.train:
            params = {
                'load': args.load,
                'display': args.show,
                'heuristic': args.heuristic,
                'quit_mode': args.mode,
                'train_steps': args.round,
            }
        else:
            params = {}
        Q.__init__(
            self,
            state_set=self._state_set,
            action_set=self._action_set,
            start_at=self._init_state_ind,
            end_at=self._end_state_ind,
            available_actions=self.available_actions,
            reward_func=self.reward,
            transition_func=self.transfer,
            run=self.run,
            q_file=Q_file,
            config_file=config,#args.config_file,
            custom_params=self.custom_params,
            sleep_time=1/speed,
            **params
        )

    def reward(self, state, action):
        rew = super().check_win(state=state, direction=action)
        if rew:
            return rew
        else:
            return self._path_cost

    def transfer(self, state, action):
        next_state = super().move(state=state, direction=action)
        return next_state

    def available_actions(self, state):
        return self._action_set

    def run(self, choose_optimal_action):
        state = self._init_state
        while True:
            action = choose_optimal_action(state=state)
            next_state = self.transfer(state=state, action=action)
            self.print_map(state=state)
            if super().check_win(state=state, direction=action):
                self.print_map(state=next_state)
                break
            state = next_state

if __name__ == '__main__':
    agent = Adaptor()
    agent.train()
