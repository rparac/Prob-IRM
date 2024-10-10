#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

# TODO: seeds
# seeds=(0 100 200 300 400)
num_agents=(1 2 10)
noise_levels=(1 0.9979081153869629 0.995305061340332 0.9814815521240234)

nodes=1
ncpus=64
ram=128

directory="recurrent_deliver_coffee"
for num_agent in "${num_agents[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="${directory}_${num_agent}_${noise_level}"
    # run noise on all three
    python submit_rcs_old_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
     ray_tests/hydra_RM_learning_PPO.py run.num_agents=${num_agent} run.should_tune=False \
      run.recurrent=True model=recurrent ppo=recurrent +hyperparams/recurrent=config1 \
      +experiment=vanilla_coffee_symmetric_error x=${noise_level}
  done
done

# Running on login.hx1.hpc.ic.ac.uk
