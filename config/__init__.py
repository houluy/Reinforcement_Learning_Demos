env_name = "TreasureHunt"

env_conf = {
    "size": 5,
}

train_render = False
evaluate_render = True

# List of algorithms that are used to train

algorithms = ["Q_learning", "SARSA_lambda"]
metrics = ["episode_number", "conv"]#"q_sum", "q_average"]

learning_rate = 0.01

epsilon_base = 0.1

gamma = 1

lmd = 0.9  # For lambda-return

evaluation_objective = "learning_rate"
evaluation_number = 2
evaluation_step = 0.01
evaluation_range = [learning_rate + evaluation_step * i for i in range(evaluation_number)]

