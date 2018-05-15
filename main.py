import Q.ProactiveCache
import Q.TreasureHunt
import matplotlib.pyplot as plt
import pdb, traceback, sys

class OutOfRangeException(Exception):
    pass

pc = Q.ProactiveCache.Adaptor()

#print(pc.u*pc.occupation(state=(0, 1), action=(1, 1)))


#print(pc.Q.q_table.ix[1, (0,0)])
try:
    conv = pc.train(conv=True, heuristic=False)
    heu_conv = pc.train(conv=True, heuristic=True)
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
#with open('results.csv', 'w') as f:
#    [f.write('{},'.format(x)) for x in conv]
#    f.write('\n')
#    [f.write('{},'.format(x)) for x in heu_conv]
#pc.train()
print(pc.Q.q_table.mean().mean())
plt.plot(range(len(conv)), conv, label='normal')
plt.plot(range(len(heu_conv)), heu_conv, label='heuristic')
plt.legend()
plt.show()
print(pc.time_ell((2, 2))*(10**-3))
print(pc.time_ell((1, 1))*(10**-3))
print(pc.time_ell((0, 0))*(10**-3))
print(pc.u*pc.occupation(state=(0, 1), action=(1, 1)))
print(pc.u*pc.occupation(state=(0, 2), action=(1, 1)))
print(pc.u*pc.occupation(state=(2, 2), action=(0, 0)))
##th = Q.TreasureHunt.Adaptor()
##conv_heu = th.train()
##conv_nor = th.train(heuristic=False)
#
##plt.plot(range(len(conv_nor)), conv_nor, label='normal')
##plt.plot(range(len(conv_heu)), conv_heu, label='heuristic')

