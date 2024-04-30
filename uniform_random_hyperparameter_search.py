import os
import random
import subprocess
import runpy

optimizer_kws_lr = [1e-1, 1e-2, 1e-3, 1e-4]
policy_train_freq = [1, 4, 8, 16, 32]
target_update_freq = list(range(500, 5000, 500))
lstm_hidden_state = [4, 8, 16, 32]
embedding_num_layers = [1, 2, 4]
embedding_layer_size = [8, 16]
embedding_output_size = [4, 8, 16]
exploration_rate_annealing_duration = list(range(5000, 100000, 1000))

random.seed(123)

num_experiments = 1  # 20

for i in range(num_experiments):
    lr = random.choice(optimizer_kws_lr)
    p_train_freq = random.choice(policy_train_freq)
    t_update_freq = random.choice(target_update_freq)
    l_hidden_state = random.choice(lstm_hidden_state)
    e_num_layers = random.choice(embedding_num_layers)
    e_layer_size = random.choice(embedding_layer_size)
    e_output_size = random.choice(embedding_output_size)
    ex_rate_annealing_duration = random.choice(exploration_rate_annealing_duration)

    directory = "test_dqrm"
    name = f"{directory}_run_{i}"
    parameters = [
        f"optimizer_kws.lr={lr}",
        f"policy_train_freq={p_train_freq}",
        f"target_update_freq={t_update_freq}",
        f"lstm_hidden_state={l_hidden_state}",
        f"embedding_num_layers={e_num_layers}",
        f"embedding_layer_size={e_layer_size}",
        f"embedding_output_size={e_output_size}",
        f"exploration_rate_annealing_duration={ex_rate_annealing_duration}",
    ]

    subprocess.run(["python", "submit_rcs_script.py", directory, name, "test_dqrn_hydra.py", *parameters])
    # runpy.run_path("submit_rcs_script.py")
