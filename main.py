import Q.ProactiveCache
import Q.TreasureHunt
import matplotlib.pyplot as plt

#pc = Q.ProactiveCache.Adaptor()
#pc.train()

th = Q.TreasureHunt.Adaptor()
conv_heu = th.train()
conv_nor = th.train(heuristic=False)

plt.plot(range(len(conv_nor)), conv_nor, label='normal')
plt.plot(range(len(conv_heu)), conv_heu, label='heuristic')
plt.legend()
plt.show()