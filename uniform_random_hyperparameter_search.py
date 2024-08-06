import os
import random
import subprocess
import runpy

# optimizer_kws_lr = [1e-1] - did not seem too important
optimizer_kws_lr = [1e-1, 1e-2, 1e-3, 1e-4]
# policy_train_freq = [1, 4, 8, 16, 32] - did not seem too
# target_update_freq = list(range(500, 5000, 500)) - ~3000 range
# lstm_hidden_state = [4, 8, 16, 32] - smaller -> better
# embedding_num_layers = [1, 2, 4] -> [1]
# embedding_layer_size = [8, 16] -> [16]
# embedding_output_size = [4] -> [4]
exploration_rate_annealing_duration = [50000, 100000, 200000, 400000]
# exploration_rate_annealing_duration = [70000] - seemed important; should try a lot larger
# buffer_sizes = [2000] - did not look too important
# use_gradient_clipping = [False]
# rho = [0.8] - [0.9]
er_batch_size = [1, 4, 16, 32]
# er_batch_size = [16] - looked useful - try different values
# er_start_size = [1000] - 1000 is fine

seeds = [123, 233, 333, 433, 533]

random.seed(123)

num_experiments = 20

for i in range(num_experiments):
    lr = random.choice(optimizer_kws_lr)
    expl_rate = random.choice(exploration_rate_annealing_duration)
    batch_size = random.choice(er_batch_size)
# for i, seed in enumerate(seeds):
    directory = "test_dqrm"
    name = f"{directory}_run_{i}"
    parameters = [
        f"optimizer_kws.lr={lr}",
        f"exploration_rate_annealing_duration={expl_rate}",
        f"er_batch_size={batch_size}",
    ]

    subprocess.run(["python", "submit_rcs_script.py", directory, name, "test_dqrn_hydra.py", *parameters])
    # runpy.run_path("submit_rcs_script.py")
