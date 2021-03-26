import matplotlib.pyplot as plt
import pathlib
import numpy as np

results_path = pathlib.Path("results")
fig_path = pathlib.Path("figs")

algorithms = ["Q_learning", "SARSA", "Average_SARSA", "SARSA_lambda"]
env = "TreasureHunt1D"

stage = "train"

typ = "Q_convergence"

fig = plt.figure()
for algo in algorithms:
    filename = f"{env}-{algo}-{stage}-{typ}.txt"
    data = np.loadtxt(results_path / filename)
    plt.plot(data, label=algo)

plt.legend()
plt.savefig(fig_path / typ)

