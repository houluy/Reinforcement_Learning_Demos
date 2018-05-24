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

# from colorline import cprint

#oprint = functools.partial(cprint, color='b', bcolor='k', end='')
#tprint = functools.partial(cprint, color='r', bcolor='k', end='')

oprint = functools.partial(print, end='')
tprint = functools.partial(print, end='')

class TreasureHunt:
    def __init__(self, speed, size):
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
            'show': None, #self.print_map,
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

class Adaptor(TreasureHunt, Q):
    def __init__(self, size=10, speed=0.5):
        TreasureHunt.__init__(self, speed, size)
        Q.__init__(
            self,
            state_set=self._state_set,
            action_set=self._action_set,
            available_actions=self.available_actions,
            reward_func=self.reward,
            transition_func=self.transfer,
            run=self.play,
            q_file='TreasureQ.csv',
            load_q=False,
            config_file=config,
            display=False,
            custom_params=self.custom_params
        )

    def reward(self, state, action):
        if super().check_win(state=state, direction=action):
            return self._instant_reward
        else:
            return 0

    def transfer(self, state, action):
        next_state = super().move(state=state, direction=action)
        return next_state

    def available_actions(self, state):
        if state == self._state_set[0]:
            return [self.right]
        #elif state == self._state_set[-1]:
        #    return [self.left]
        else:
            return self._action_set

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

if __name__ == '__main__':
    agent = Adaptor()
    agent.train()
