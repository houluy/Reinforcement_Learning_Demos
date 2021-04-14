env_name = "TreasureHunt"

env_conf = {
    "size": 7,#(5, 5),
}

train_render_config = {
    "train_render": False,
    "train_render_interval": 0,
    "train_render_clear": False,  # Whether to clear screen before render
}
evaluate_render = True

train_termination_config = {
    # When to terminate training, by number of episodes or by loss
    "termination_type": "loss", #"episode_num",  # "loss"
    "termination_precision": 1e-4,  # Only used for convergence by loss,
}

max_train_episodes = 1000
info_episodes_pro = 0.2  # Percentage. Output info of training process after info_episodes_pro * max_train_episodes episodes.
info_episodes = 20  #int(info_episodes_pro * max_train_episodes)  # Or directly set the info episode

init_q_mode = "random"  # Methods to initialize values of Q table. Can be one of ["random", "zero", "positive", "negetive", "custom"]

# List of algorithms that are used to train

algorithms = ["Q_lambda", "SARSA_lambda", "Q_learning", "SARSA", "Average_SARSA"]
metrics = ["episode_total_reward", "q_sum"]#, "q_average"]

learning_rate = 0.1
epsilon_base = 0.5
epsilon_decay_rate = 0.99

gamma = 1
lmd = 0.9  # For lambda-return

#evaluation_objective = "gamma"
#evaluation_number = 5
#evaluation_start = 1
#evaluation_step = 0.05
#order = -1
#objective_values = [
#    evaluation_start + order * evaluation_step * i for i in range(evaluation_number) 
#]
evaluation_objective = "learning_rate"
#evaluation_number = 4
#evaluation_step = 0.05
#order = 1

#evaluation_objective = "epsilon_decay_rate"
evaluation_number = 1
evaluation_start = 0.1
evaluation_step = 0.1
order = 1

objective_values = [evaluation_start + order * evaluation_step * i for i in range(evaluation_number)]

