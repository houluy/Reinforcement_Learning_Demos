import matplotlib.pyplot as plt
import pathlib
import numpy as np


class Results:
    def __init__(self):
        pass

    def transform_results(self):
        """ Transform results from algorithm based to metric based """
        self.m_results = {
            k: {
                algo: 0.0 for algo in self.algorithms     
            } for k in self.metrics
        }
        
        for m in self.metrics:
            for algo in self.algorithms:
                self.m_results[m][algo] = self.results[algo][m]


class Visualization:
    def __init__(self, results):
        self.fig_path = pathlib.Path("figs")
        self.results = results
        self.algorithms = list(self.results.keys())
        self.target_values = list(self.results[self.algorithms[0]].keys())
        self.metrics = list(self.results[self.algorithms[0]][self.target_values[0]].keys())
   
    def each_algorithm(self, metric_name="conv"):
        fig = plt.figure()
        for algo, target in self.results.items():
            for value, mtcs in target.items():
                plt.plot(mtcs[metric_name], label=f"{algo}-{value}")
        plt.title(f"{metric_name} of different algorithms")
        plt.xlabel("Number of episodes")
        plt.ylabel(f"Value of {metric_name}")
        plt.legend()
        plt.savefig(self.fig_path / metric_name)

    def each_metric(self):
        pass



#results_path = pathlib.Path("results")
#fig_path = pathlib.Path("figs")
#
#algorithms = ["Q_learning", "SARSA", "Average_SARSA", "SARSA_lambda"]#, "Q_lambda"]
##algorithms = ["Q_learning"]
#env = "TreasureHunt2D"
#
#stage = "train"
#
#typ = "Q_convergence"
#
#fig = plt.figure()
#for algo in algorithms:
#    filename = f"{env}-{algo}-{stage}-{typ}.txt"
#    data = np.loadtxt(results_path / filename)
#    plt.plot(data, label=algo)
#
#plt.legend()
#plt.savefig(fig_path / typ)
#
