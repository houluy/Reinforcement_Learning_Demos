env_name = "TreasureHunt2D"

env_conf = {
    "size": (4, 4),
}

train_render = False
evaluate_render = True

# List of algorithms that are used to train

algorithms = ["Q_lambda", "SARSA_lambda"]
metrics = ["q_sum"]#"q_sum", "q_average"]

learning_rate = 0.1

epsilon_base = 0.5
epsilon_decay_rate = 0.9

gamma = 1

lmd = 0.9  # For lambda-return

episodes = 1000

#evaluation_objective = "gamma"
#evaluation_number = 5
#evaluation_start = 1
#evaluation_step = 0.05
#order = -1
#objective_values = [
#    evaluation_start + order * evaluation_step * i for i in range(evaluation_number) 
#]
#evaluation_objective = "learning_rate"
#evaluation_number = 4
#evaluation_step = 0.05
#order = 1

evaluation_objective = "epsilon_decay_rate"
evaluation_number = 1
evaluation_start = 1
evaluation_step = 0.1
order = -1

objective_values = [evaluation_start + order * evaluation_step * i for i in range(evaluation_number)]

