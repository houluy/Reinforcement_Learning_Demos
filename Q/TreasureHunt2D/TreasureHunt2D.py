import pandas as pd
import numpy as np
import random
import pathlib
df = pd.DataFrame
from Q.Q.Q import Q
import functools
from itertools import product

file_path = pathlib.Path(__file__).parent
config = file_path / '2dtreasure.yaml'
Q_file = file_path / '2dTreasureQ.csv'

from colorline import cprint

oprint = functools.partial(cprint, color='b', bcolor='k', end='')
trapprint = functools.partial(cprint, color='r', bcolor='k', end='')
wprint = functools.partial(cprint, color='y', bcolor='k', end='')
tprint = functools.partial(cprint, color='c', bcolor='k', end='')
bprint = functools.partial(print, end='')
eprint = functools.partial(cprint, color='g', bcolor='k', end='')

def gen_randmap(size):
    randmap = df(
        np.zeros(size)
    )
    all_coors = [(x, y) for x, y in product(range(size[0]), range(size[1]))]
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
        randmap,
        trap_points,
        wall_points,
        treasure_points,
        all_coors,
    )

def add_tuple(a, b):
    return tuple(sum(x) for x in zip(a, b))


class TreasureHunt2D:

    def check_pos(func):
        def wrapper(self, pos=None, *args, **kwargs):
            if not pos:
                pos = self._warrior_pos
            return func(self, pos=pos, *args, **kwargs)
        return wrapper
    
    def __init__(self, size, warrior_ch='@', dest_ch='#', trap_ch='X', wall_ch='+', blank_ch=' '):
        self._size = size
        self._maps, self._trap, self._wall, self._treasure, self._path = gen_randmap((self._size,)*2)
        self._terminal_points = self._trap.append(self._treasure)
        self._warrior_pos = (0, 0)
        self._warrior_ch = warrior_ch
        self._occupation = 0
        self._maps.iat[self._warrior_pos] = 3
        self._dest_ch = dest_ch
        self._trap_ch = trap_ch
        self._wall_ch = wall_ch
        self._blank_ch = blank_ch
        self._rep = [-1, 0, 1, 2, 3]
        self._char = [self._trap_ch, self._blank_ch, self._wall_ch, self._dest_ch, self._warrior_ch]
        self._printfunc = [trapprint, bprint, wprint, tprint, eprint]
        self._char_map = dict(zip(self._rep, self._char))
        self._print_map = dict(zip(self._rep, self._printfunc))
        self._win_pos = ((self._size - 1,)*2)
        self._direction = {
            (0, 1): '↓',
            (1, 0): '→',
            (0, -1): '↑',
            (-1, 0): '←',
        }

    def display(self):
        for x, row in self._maps.iterrows():
            print('|', end='')
            for y in row:
                print_func = self._print_map.get(y)
                print_func(self._char_map.get(y))
                print('|', end='')
            print()
        print()

    def check_boundary(self, pos):
        for c in pos:
            if c < 0 or c >= self._size:
                return True
        return False

    @check_pos
    def check_win(self, pos):
        if pos in self._terminal_points:
            return True
        return False
    
    @check_pos
    def available_moves(self, pos):
        all_moves = self._direction.keys()
        amoves = []
        for move in all_moves:
            npos = add_tuple(self._warrior_pos, move)
            if not (npos in self._wall or self.check_boundary(npos)):
                amoves.append(move)
        return amoves

    @check_pos
    def move(self, direction, pos):
        assert direction in self._direction.keys()
        #self._maps.iat[self._warrior_pos] = 0
        self._warrior_pos = add_tuple(self._warrior_pos, direction)
        self._maps.iat[self._warrior_pos] = 3


class T2DAdaptor(TreasureHunt2D, Q):
    def __init__(self, size, args=None):
        TreasureHunt2D.__init__(self, size)
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
    m = th2d.available_moves()
    #print(m)
    th2d.move(direction=m[0])
    th2d.display()
