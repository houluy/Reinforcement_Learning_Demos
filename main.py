import Q.TreasureHunt.TreasureHunt as TreasureHunt
import Q.TreasureHunt2D.TreasureHunt2D as T2D

import matplotlib.pyplot as plt
import argparse
import pdb, traceback, sys

parser = argparse.ArgumentParser(description='This is a demo to show how Q_learning makes agent intelligent')

mode_parser = parser.add_subparsers(title='mode', help='Choose a mode')

train_parser = mode_parser.add_parser('train', help='Train an agent')
run_parser = mode_parser.add_parser('run', help='Make an agent run')

# Arguments for training
train_parser.add_argument('-m', '--mode', help='Training mode, by rounds or by convergence', choices=['c', 'r'], default='c')
train_parser.add_argument('-r', '--round', help='Training rounds, neglect when convergence is chosen', default=300, type=int)
train_parser.add_argument('-l', '--load', help='Whether to load Q table from a csv file when training', action='store_true', default=False)
train_parser.add_argument('-s', '--show', help='Show the training process.', action='store_true', default=False)
train_parser.add_argument('-c', '--config_file', help='Config file for significant parameters', default=None)
train_parser.add_argument('-d', '--demo', help='Choose a demo to run', choices=['t'], default='t')
train_parser.add_argument('-a', '--heuristic', help='Whether to use a heuristic iteration', action='store_true', default=False)

# Arguments for running
run_parser.add_argument('-d', '--demo', help='Choose a demo to run', choices=['t'], default='t')
run_parser.add_argument('-q', help='Choose a Q table from a csv file', type=argparse.FileType('r'))


# def train(args):
#     args.train = True
# 
# def run(args):
#     args.train = False
# 
# train_parser.set_defaults(func=train)
# run_parser.set_defaults(func=run)
# 
# args = parser.parse_args()
# args.func(args)
# try:
#     th = TreasureHunt.Adaptor(args=args)
#     th.start(mode=args.train)
# except:
#     type, value, tb = sys.exc_info()
#     traceback.print_exc()
#     pdb.post_mortem(tb)

t = T2D.T2DAdaptor(10)
t.display()
m = t.available_moves()
t.move(direction=m[0])
t.display()
