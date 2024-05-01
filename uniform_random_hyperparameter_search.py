import os
import random
import subprocess
import runpy

# optimizer_kws_lr = [1e-1]
# policy_train_freq = [1, 4, 8, 16, 32]
# target_update_freq = list(range(500, 5000, 500))
# lstm_hidden_state = [4, 8, 16, 32]
# embedding_num_layers = [1, 2, 4]
# embedding_layer_size = [8, 16]
# embedding_output_size = [4]# [4, 8, 16]
# exploration_rate_annealing_duration = [70000]
buffer_sizes = [500, 1000, 2000, 4000, 10000]
use_gradient_clipping = [True, False]
rho = [0.8, 0.9, 0.95, 0.99]
er_batch_size = [1, 16, 32]
er_start_size = [1, 100, 1000]

random.seed(123)

num_experiments = 1  # 20

for i in range(num_experiments):
    buffer_size = random.choice(buffer_sizes)
    u_gradient_clipping = random.choice(use_gradient_clipping)
    r = random.choice(rho)
    e_start_size = random.choice(er_start_size)
    e_batch_size = random.choice(er_batch_size)

    directory = "test_dqrm"
    name = f"{directory}_run_{i}"
    parameters = [
        f"buffer_size={buffer_size}",
        f"use_gradient_clipping={u_gradient_clipping}",
        f"optimizer_kws.rho={r}",
        f"er_start_size={e_start_size}",
        f"er_batch_size={e_batch_size}",
    ]

    subprocess.run(["python", "submit_rcs_script.py", directory, name, "test_dqrn_hydra.py", *parameters])
    # runpy.run_path("submit_rcs_script.py")
