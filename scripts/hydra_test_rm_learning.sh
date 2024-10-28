#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

nodes=1
ncpus=64
ram=128 # Gb

directory="deliver_coffee"
# name="rm_learning_experiment_tuning"
name="rm_learning_experiment_with_params"

python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/hydra_RM_learning_PPO.py run.use_perfect_rm=False run.num_agents=2 run.should_tune=False +hyperparams/with_rm=config1
