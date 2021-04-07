import matplotlib.pyplot as plt
import pathlib 
import numpy as np


class Result:
    def __init__(
        self,
        algorithms: list,
        evaluation_objective: str,
        objective_values: list,
        metrics: list,
        episodes: int
    ):
        """ Wrap the results 
            Structure(dict-like style):
                self[algorithm][target_value][metric] = metric_value 

            Parameters:
                @algorithms: RL algorithms need to be evaluated
                @variable_name: Name of the variable.
                @variable_values: All possible values of the variable
                @metrics: Name of all metrics
        """
        self.algorithms = algorithms
        self.alg_len = len(self.algorithms)
        self.alg_ind = dict(zip(self.algorithms, range(self.alg_len)))
        self.evaluation_objective = evaluation_objective
        self.objective_values = objective_values
        self.obj_len = len(self.objective_values)
        self.obj_ind = dict(zip(self.objective_values, range(self.obj_len)))
        self.metrics = metrics
        self.met_len = len(self.metrics)
        self.met_ind = dict(zip(self.metrics, range(self.met_len)))
        self.episodes = episodes
        self.metric_values = np.zeros((self.alg_len, self.obj_len, self.met_len, self.episodes))
        self._short_name_dict = {
            "alg": "algorithms",
            "var": "objective_values",
            "met": "metrics",
        }

    def __getattr__(self, short_name):
        return super().__getattr__(self, self._short_name_dict[short_name])

    def reshape(self, one="alg_len", two="var_len"):
        """ Transform results from algorithm based to metric based 
            Default sequence:
                algorithms (0) → objective (1) → metrics (2)
            
            Parameters:
                @one, two: Define the axis sequence of value for better plotting
                    ("alg", "obj", "met")
        """
        total_names = frozenset(("alg", "obj", "met"))
        axis = (
            getattr(self, one),
            getattr(self, two),
            getattr(self, (total_names - {one, two}).pop()),
        )
        self.metric_values.reshape(axis)
        #self.m_results = {
        #    k: {
        #        algo: 0.0 for algo in self.algorithms     
        #    } for k in self.metrics
        #}
        #
        #for m in self.metrics:
        #    for algo in self.algorithms:
        #        self.m_results[m][algo] = self.results[algo][m]

    def _ind_to_pos(self, ind: tuple):
        algorithm, objective, metric = ind
        pos = (
            self.alg_ind[algorithm],
            self.obj_ind[objective],
            self.met_ind[metric],
        )
        return pos

    def __getitem__(self, ind: tuple):
        return self.metric_values[self._ind_to_pos(ind)]

    def __setitem__(self, ind: tuple, value: float):
        self.metric_values[self._ind_to_pos(ind)] = value

    def __str__(self):
        ret = []
        nl = '\n'
        for alg in self.algorithms:
            alg_str_lst = []
            for obj in self.objective_values:
                obj_str_lst = []
                for met in self.metrics:
                    met_str = f"{met} = {self[(alg, obj, met)]}"
                    obj_str_lst.append(met_str)
                obj_str = f"{self.evaluation_objective} = {obj}{nl}{nl.join(obj_str_lst)}"
                alg_str_lst.append(obj_str)
            alg_str = f"Algorithm: {alg}:{nl}{nl.join(alg_str_lst)}"
            ret.append(alg_str)
        return '\n'.join(ret)


class Visualization:
    def __init__(self, result):
        self.fig_path = pathlib.Path("figs")
        self.result = result
   
    def plot_by_metric(self, metric="q_sum"):
        fig = plt.figure()
        for alg in self.result.algorithms:
            for obj in self.result.objective_values:
                plt.plot(self.result[(alg, obj, metric)], label=f"{alg}-{self.result.evaluation_objective}={obj}")
        plt.title(f"{metric} of different algorithms include {' '.join(self.result.algorithms)}\n"
            f"with {self.result.evaluation_objective} ranges\n"
            f"in [{self.result.objective_values[0]}, {self.result.objective_values[-1]}]")
        plt.xlabel("Number of episodes")
        plt.ylabel(f"Value of {metric}")
        plt.legend()
        plt.savefig(self.fig_path / metric)


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
