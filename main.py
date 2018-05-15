import Q.ProactiveCache
import Q.TreasureHunt
import matplotlib.pyplot as plt
import pdb, traceback, sys

pc = Q.ProactiveCache.Adaptor()
#print(pc.Q.q_table.ix[1, (0,0)])
try:
    conv = pc.train(heuristic=False)
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
#pc.train()

#th = Q.TreasureHunt.Adaptor()
#conv_heu = th.train()
#conv_nor = th.train(heuristic=False)

#plt.plot(range(len(conv_nor)), conv_nor, label='normal')
#plt.plot(range(len(conv_heu)), conv_heu, label='heuristic')
#plt.legend()
#plt.show()
