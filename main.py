import Q.ProactiveCache
import Q.TreasureHunt
import matplotlib.pyplot as plt
import pdb, traceback, sys

class OutOfRangeException(Exception):
    pass

pc = Q.ProactiveCache.Adaptor()

#print(pc.Q.q_table.ix[1, (0,0)])
try:
    conv = pc.train(conv=False, heuristic=False)
    #heu_conv = pc.train(conv=False, heuristic=True)
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
#with open('results.csv', 'w') as f:
#    [f.write('{},'.format(x)) for x in conv]
#    f.write('\n')
#    [f.write('{},'.format(x)) for x in heu_conv]
#pc.train()
plt.plot(range(len(conv)), conv, label='normal')
##plt.plot(range(len(heu_conv)), heu_conv, label='heuristic')
plt.legend()
plt.show()
##th = Q.TreasureHunt.Adaptor()
##conv_heu = th.train()
##conv_nor = th.train(heuristic=False)
#
##plt.plot(range(len(conv_nor)), conv_nor, label='normal')
##plt.plot(range(len(conv_heu)), conv_heu, label='heuristic')

