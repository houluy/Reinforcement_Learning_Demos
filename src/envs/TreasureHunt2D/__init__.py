import pandas as pd
import numpy as np
import random
import pathlib
import yaml
df = pd.DataFrame
import functools
import time
from itertools import product
from collections import OrderedDict
from pprint import pprint
from math import sqrt
import pdb

file_path = pathlib.Path(__file__).parent
config_file = file_path / 'TreasureHunt2D.yml'
Q_file = file_path / 'Treasure2DQ.csv'
conv_file = file_path / 'Treasure2Dconv.csv'
mapfile = file_path / 'Treasure2Dmap.csv'

from colorline import cprint

oprint = functools.partial(cprint, color='b', bcolor='k', end='')
trapprint = functools.partial(cprint, color='r', bcolor='k', end='')
wprint = functools.partial(cprint, color='y', bcolor='k', end='')
tprint = functools.partial(cprint, color='c', bcolor='k', end='')
bprint = functools.partial(print, end='')
eprint = functools.partial(cprint, color='g', bcolor='k', end='')
hprint = functools.partial(cprint, color='p', bcolor='k', end='')


def add_tuple(a, b):
    return tuple(sum(x) for x in zip(a, b))


class TreasureHunt2D:
    @staticmethod
    def gen_randmap(size):
        randmap = df(
            np.zeros(size),
            dtype=np.int32
        )
        all_coordinates = [(x, y) for x, y in product(range(size[0]), range(size[1]))]
        cp_allcs = all_coordinates[:]
        treasure_point = all_coordinates.pop()  # Treasure point at the bottom-right
        all_coordinates.pop(0)  # Warrior at top-left
        randmap.at[treasure_point] = 2
        wall_count = min(size) - 3
        trap_count = wall_count + 1
        trap_points = []
        wall_points = []
        for x in range(wall_count + trap_count):
            coor = random.choice(all_coordinates)
            if x < wall_count:
                wall_points.append(coor)
                randmap.at[coor] = 1
            else:
                trap_points.append(coor)
                randmap.at[coor] = -1

        return (
            cp_allcs,
            randmap,
            trap_points,
            wall_points,
            treasure_point,
            all_coordinates,
        )

    @staticmethod
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
        return [maps.shape[0], coors, *c_dic.values()]

    def check_pos(func):
        def wrapper(self, pos=None, *args, **kwargs):
            if not pos:
                pos = self.observation
            return func(self, pos=pos, *args, **kwargs)
        return wrapper
       
    def __init__(self, mapfile=None, size=(5, 5), warrior_ch='@', dest_ch='#', trap_ch='X', wall_ch='-', blank_ch=' '):
        if (mapfile is None) or (not pathlib.Path(mapfile).exists()):
            self.size = size
            self.all_coordinates, self.maps, self.trap, self.wall, self.treasure, self.path = self.gen_randmap(self.size)
            self.observation = (0, 0) #random.choice(self._path)
            self.save_map()
        else:
            self.maps = self.load_map(mapfile)
            self.size, self.all_coordinates, self.trap, self.path, self.wall, self.treasure, self.observation = self.rec_randmap(self.maps)
            self.treasure = self.treasure[0]
            self.observation = (0, 0)
        self.name = "TreasureHunt2D"
        self.terminal_points = self.trap + [self.treasure]
        self.run_sleep = 0.1
        self.warrior_ch = warrior_ch
        self.dest_ch = dest_ch
        self.trap_ch = trap_ch
        self.wall_ch = wall_ch
        self.blank_ch = blank_ch
        self.occupation = 0
        self.points = [-1, 0, 1, 2, 3]
        self.history_path = []#[self.observation]
        self.char = [self.trap_ch, self.blank_ch, self.wall_ch, self.dest_ch, self.warrior_ch]
        self.printfunc = [trapprint, bprint, wprint, tprint, eprint]
        self.char_map = dict(zip(self.points, self.char))
        self.print_map = dict(zip(self.points, self.printfunc))
        self.observation_space = self.all_coordinates
        self.action_space = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.directions_str = ['↓', '→', '↑', '←']
        self.direction = dict(zip(self.action_space, self.directions_str))
        #TODO
        self.defaultrewards = [-10, -0.01, None, 10, None]
        self.reward_dic = dict(zip(self.points, self.defaultrewards))

    @staticmethod
    def load_map(mapfile=mapfile):
        maps = pd.read_csv(mapfile, index_col=0)
        maps.columns = maps.columns.astype(int)
        return maps

    def save_map(self, mapfile=mapfile):
        self.maps.to_csv(mapfile)

    def test(self):
        #pdb.set_trace()
        self.render()
        amoves = self.available_moves(pos=(0, 1))
        print(amoves)

    def check_boundary(self, pos):
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.size[0] or pos[1] >= self.size[1]:
            return True
        return False

    @check_pos
    def __getitem__(self, pos):
        return self.maps.iat[pos]

    @check_pos
    def check_win(self, pos):
        if pos == self.treasure:
            return True
        elif pos in self.trap:
            return False
        return None

    @check_pos
    def check_win_by_action(self, pos, direction):
        npos = self.move(direction=direction, pos=pos)
        return self.check_win(pos=npos)
 
    def action_filter(self, state):
        all_moves = self.action_space
        amoves = []
        for move in all_moves:
            npos = add_tuple(state, move)
            if not (npos in self.wall or self.check_boundary(npos)):
                amoves.append(move)
        return amoves

    def move(self, direction):
        pos = add_tuple(self.observation, direction)
        return pos

    def step(self, action):
        self.history_path.append(self.observation)
        self.maps.at[self.observation] = 0
        self.observation = self.move(direction=action)
        #self.maps.at[self.observation] = 1
        #print(self.observation)
        reward = self.reward_dic[self.maps.at[self.observation]]
        done = False
        if self.observation in self.terminal_points:
            done = True
        return self.observation, reward, done, {}

    def reset(self):
        self.observation = (0, 0)
        self.history_path.clear()
        return self.observation

    def render(self):
        pos = self.observation
        for x, row in self.maps.iterrows():
            print('|', end='')
            for y, col in row.iteritems():
                if (x, y) == pos:
                    eprint(self.warrior_ch)
                elif (x, y) in self.history_path:
                    hprint(self.warrior_ch)
                else:
                    self.print_map.get(col)(self.char_map.get(col))
                print('|', end='')
            print()
        print()

    def close(self):
        pass

    def seed(self):
        pass

    def sample_run(self):
        done = False
        state = self.reset()
        self.render()
        while not done:
            time.sleep(self.run_sleep)
            action = random.choice(self.action_filter(state))
            next_state, reward, done, info = self.step(action=action)
            self.render()
            if done:
                print("Done!")
            state = next_state


    @check_pos
    def Euclidean(self, pos):
        return sqrt((pos[0] - self.treasure[0])**2 + (pos[1] - self.treasure[1])**2)

    @check_pos
    def Manhattan(self, pos):
        return abs(pos[0] - self.treasure[0]) + abs(pos[1] - self.treasure[1])


#class Adaptor(TreasureHunt2D, Agent):
#    def __init__(self, params=None, **kwargs):
#        kwargs['mapfile'] = mapfile
#        config = yaml.load(open(config_file))
#        config2D = config.get('2D')
#        kwargs['size'] = config2D.get('size')
#        speed = config2D.get('speed')
#        TreasureHunt2D.__init__(self, **kwargs)
#        self._state_space = self._all_coors
#        self._action_space = self._all_dirs
#        self._current_state = self._warrior_pos
#        self._defaultrewards = [-10, -0.5, None, 10, None]
#        self._reward_dic = dict(zip(self._points, self._defaultrewards))
#        self._custom = {
#            'show': self.render,
#        }
#        Q_params = config.get('Q')
#        Agent.__init__(
#            self,
#            state_set=self._state_space,
#            action_set=self._action_space,
#            start_states=[(0, 0)],#self._path,
#            end_states=self._terminal_points,
#            init=self.init,
#            ahook=self.save_map,
#            available_actions=self.action_filter,
#            reward_func=self.reward,
#            transition_func=self.transfer,
#            q_file=Q_file,
#            conv_file=conv_file,
#            run=self.run,
#            sleep_time=speed,
#            custom_params=self._custom,
#            **params,
#            **Q_params
#        )
#
#    def reward(self, state, action):
#        npos = self.move(direction=action, pos=state)
#        return self._reward_dic.get(self[npos])
#
#    def heuristic_reward(self, state, action):
#        dist_bef = self.Manhattan(pos=state)
#        npos = self.move(direction=action, pos=state)
#        dist_aft = self.Manhattan(pos=npos)
#        ir = self._reward_dic.get(self[npos])
#        if (ir == 10) or (ir == -10):
#            return ir
#        else:
#            return ir + 0.1*(dist_bef - dist_aft)
#
#    def available_actions(self, state):
#        return self.available_moves(pos=state)
#
#    def render(self, state):
#        super().render(pos=state)
#
#    def reset(self):
#        self.warrior_pos = (0, 0)
#        self.history_path = []
#
#    
#if __name__ == '__main__':
#    th2d = TreasureHunt2D(10)
#    th2d.render()
#
#    #print(m)
#    #th2d.move(direction=m[0])
#    th2d.update_map(direction=m[0])
#    th2d.render()
