#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

nodes=2
ncpus=64
ram=128 # Gb

directory="deliver_coffee"
name="rm_learning_experiment"

python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/hydra_RM_learning_PPO.py run.num_agents=2 run.use_perfect_rm=False run.should_tune=False run.use_rs=True run.stop_iters=100 run.seed=127 +hyperparams/with_rm=andrew +experiment=vanilla_coffee_symmetric_error x=0.9814815521240234

