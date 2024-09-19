#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

seeds=(0) # 100 200 300 400)
noise_levels=(1) # 0.9979081153869629 0.995305061340332 0.9814815521240234)

directory="deliver_coffee"
for seed in "${seeds[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="${directory}_${seed}_${noise_level}"
    # run noise on all three
    python submit_rcs_script.py ${directory} ${name} \
     ray_tests/RM_learning_PPO.py --enable-new-api-stack --wandb-project=prob-irm --stop-iters=25  \
      --use-perfect-rm --wandb-run-name=single_agent_rm \
      --wandb-key=INSERT
  done
done

# Running on login.hx1.hpc.ic.ac.uk
