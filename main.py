import Q.TreasureHunt.TreasureHunt as TreasureHunt

import matplotlib.pyplot as plt
import argparse
import pdb, traceback, sys

parser = argparse.ArgumentParser(description='This is a demo to show how Q_learning makes agent intelligent')

mode_parser = parser.add_subparsers(title='mode', help='Choose a mode')

train_parser = mode_parser.add_parser('train', help='Train an agent')
run_parser = mode_parser.add_parser('run', help='Make an agent run')

parser.add_argument('-l', '--load', help='Whether to load Q table from a csv file', action='store_true')
parser.add_argument('-d', '--demo', help='Choose a demo to run', choices=['t'], default='t')
parser.add_argument('-s', '--show', help='Show the training process.', action='store_true', default=False)
parser.add_argument('-c', '--config_file', help='Config file for significant parameters', default=None)
train_parser.add_argument('-m', '--mode', help='Training mode, by rounds or by convergence', choices=['c', 'r'], default='c')
train_parser.add_argument('-r', '--round', help='Training rounds, neglect when convergence is chosen', default=300)

def args_train(args):
    pass

def args_run(args):
    pass

args = parser.parse_args()
#train_args = train_parser.parse_args()
print(args)
#print(train_args)

#try:
#    th = TreasureHunt.Adaptor()
#    th.train()
#except:
#    type, value, tb = sys.exc_info()
#    traceback.print_exc()
#    pdb.post_mortem(tb)
