import pandas as pd
import numpy as np
import random
import pathlib
df = pd.DataFrame
from Q.Q.Q import Q
import functools
from itertools import product
from collections import OrderedDict
from pprint import pprint
import pdb

file_path = pathlib.Path(__file__).parent
config = file_path / 'TreasureHunt2D.yml'
Q_file = file_path / 'Treasure2DQ.csv'
mapfile = file_path / 'Treasure2Dmap.csv'

from colorline import cprint

oprint = functools.partial(cprint, color='b', bcolor='k', end='')
trapprint = functools.partial(cprint, color='r', bcolor='k', end='')
wprint = functools.partial(cprint, color='y', bcolor='k', end='')
tprint = functools.partial(cprint, color='c', bcolor='k', end='')
bprint = functools.partial(print, end='')
eprint = functools.partial(cprint, color='g', bcolor='k', end='')

def gen_randmap(size):
    randmap = df(
        np.zeros(size),
        dtype=np.int8
    )
    all_coors = [(x, y) for x, y in product(range(size[0]), range(size[1]))]
    cp_allcs = all_coors[:]
    treasure_points = all_coors.pop()
    all_coors.pop(0)
    randmap.iat[treasure_points] = 2
    wall_count = min(size) - 1
    trap_count = wall_count - 1
    trap_points = []
    wall_points = []
    for x in range(wall_count + trap_count):
        coor = random.choice(all_coors)
        if x < wall_count:
            wall_points.append(coor)
            randmap.iat[coor] = 1
        else:
            trap_points.append(coor)
            randmap.iat[coor] = -1

    return (
        cp_allcs,
        randmap,
        trap_points,
        wall_points,
        treasure_points,
        all_coors,
    )

def rec_randmap(maps):
    coors, trap, wall, treasure, path, warrior = [[] for _ in range(6)]
    c_dic = OrderedDict({
        -1: trap,
        0: path,
        1: wall,
        2: treasure,
        3: warrior,
    })
    for r_ind, r_val in maps.iterrows():
        for c_ind, c_val in r_val.iteritems():
            pos = (r_ind, c_ind)
            coors.append(pos)
            c_dic.get(c_val).append(pos)
    pprint(c_dic.values())
    return [maps.shape[0], coors, *c_dic.values()]

def add_tuple(a, b):
    return tuple(sum(x) for x in zip(a, b))

class TreasureHunt2D:

    def check_pos(func):
        def wrapper(self, pos=None, *args, **kwargs):
            if not pos:
                pos = self._warrior_pos
            return func(self, pos=pos, *args, **kwargs)
        return wrapper
       
    def __init__(self, mapfile=None, size=10, warrior_ch='@', dest_ch='#', trap_ch='X', wall_ch='-', blank_ch=' '):
        if mapfile is None:
            self._size = size
            self._all_coors, self._maps, self._trap, self._wall, self._treasure, self._path = gen_randmap((self._size,)*2)
            self._warrior_pos = (0, 0) #random.choice(self._path)
            self._maps.iat[self._warrior_pos] = 3
        else:
            self._maps = self.load_map(mapfile)
            self._size, self._all_coors, self._trap, self._path, self._wall, self._treasure, self._warrior_pos = rec_randmap(self._maps)
            self._warrior_pos = self._warrior_pos[0]
        self._terminal_points = self._trap + [self._treasure]
        self._warrior_ch = warrior_ch
        self._occupation = 0
        self._dest_ch = dest_ch
        self._trap_ch = trap_ch
        self._wall_ch = wall_ch
        self._blank_ch = blank_ch
        self._points = [-1, 0, 1, 2, 3]
        self._history_path = [self._warrior_pos]
        self._char = [self._trap_ch, self._blank_ch, self._wall_ch, self._dest_ch, self._warrior_ch]
        self._printfunc = [trapprint, bprint, wprint, tprint, eprint]
        self._char_map = dict(zip(self._points, self._char))
        self._print_map = dict(zip(self._points, self._printfunc))
        self._win_pos = ((self._size - 1,)*2)
        self._all_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self._directions_str = ['↓', '→', '↑', '←']
        self._direction = dict(zip(self._all_dirs, self._directions_str))

    def load_map(self, mapfile):
        self._maps = pd.read_csv(mapfile, index_col=0)
        self._maps.columns = self._maps.columns.astype(int)
        return self._maps

    def save_map(self, mapfile):
        self._maps.to_csv(mapfile)

    def display(self, state=None):
        for x, row in self._maps.iterrows():
            print('|', end='')
            for y, col in row.iteritems():
                if (x, y) in self._history_path:
                    print_func = eprint
                else:
                    print_func = self._print_map.get(col)
                print_func(self._char_map.get(col))
                print('|', end='')
            print()
        print()

    def check_boundary(self, pos):
        for c in pos:
            if c < 0 or c >= self._size:
                return True
        return False

    @check_pos
    def __getitem__(self, pos):
        return self._maps.iat[pos]

    @check_pos
    def check_win(self, pos):
        if pos == self._treasure:
            return True
        elif pos in self._trap:
            return False
        return None

    @check_pos
    def check_win_by_action(self, pos, direction):
        npos = self.move(direction=direction, pos=pos)
        return self.check_win(pos=npos)
    
    @check_pos
    def available_moves(self, pos):
        all_moves = self._all_dirs
        amoves = []
        for move in all_moves:
            npos = add_tuple(self._warrior_pos, move)
            if not (npos in self._wall or self.check_boundary(npos)):
                amoves.append(move)
        return amoves

    @check_pos
    def move(self, direction, pos):
        assert direction in self._all_dirs
        pos = add_tuple(pos, direction)
        return pos

    @check_pos
    def update_map(self, direction, pos):
        self._history_path.append(pos)
        self._maps.iat[pos] = 0
        self._warrior_pos = self.move(direction=direction, pos=pos)
        self._maps.iat[self._warrior_pos] = 3
        return self._warrior_pos


class Adaptor(TreasureHunt2D, Q):
    def __init__(self, params=None, load_map=False, **kwargs):
        if load_map:
            kwargs['mapfile'] = mapfile
        else:
            kwargs['size'] = 10
        TreasureHunt2D.__init__(self, **kwargs)
        self._state_space = self._all_coors
        self._action_space = self._all_dirs
        self._current_state = self._warrior_pos
        self._defaultrewards = [-10, -2, None, 10, None]
        self._reward_dic = dict(zip(self._points, self._defaultrewards))
        self._custom = {
            'show': self.display,
        }
        Q.__init__(
            self,
            state_set=self._state_space,
            action_set=self._action_space,
            start_state=self._warrior_pos,
            end_states=self._terminal_points,
            available_actions=self.available_actions,
            reward_func=self.reward,
            transition_func=self.transfer,
            config_file=config,
            q_file=Q_file,
            run=self.run,
            sleep_time=1/10,
            custom_params=self._custom,
            **params
        )

    def reward(self, state, action):
        return self._reward_dic.get(self[self.move(direction=action, pos=state)])

    def transfer(self, state, action):
        self._current_state = self.update_map(direction=action, pos=state)
        return self._current_state

    def available_actions(self, state):
        return self.available_moves(pos=state)

    def test(self):
        #pdb.set_trace()
        amoves = self.available_moves()
        for m in amoves:
            #pdb.set_trace()
            print(self.reward(self._current_state, m))

    def run(self, choose_optimal_action):
        state = self._warrior_pos
        while True:
            action = choose_optimal_action(state=state)
            nstate = self.transfer(state=state, action=action)
            self.update_map(direction=action, pos=state)
            self.display()
            winning = self.check_win()
            if winning is None:
                state = nstate
            else:
                if winning:
                    print('Finds!')
                else:
                    print('Dies!')
        #if args.train:
        #    params = {
        #        'load': args.load,
        #        'display': args.show,
        #        'heuristic': args.heuristic,
        #        'quit_mode': args.mode,
        #        'train_steps': args.round,
        #    }
        #else:
        #    params = {}
        #Q.__init__(
        #    self,



if __name__ == '__main__':
    th2d = TreasureHunt2D(10)
    th2d.display()

    #print(m)
    #th2d.move(direction=m[0])
    th2d.update_map(direction=m[0])
    th2d.display()