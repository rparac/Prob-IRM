#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

nodes=1
ncpus=64
ram=128 # Gb

directory="deliver_coffee"
name="with_hyperparam_experiment"

cmd="python submit_rcs_old_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/hydra_RM_learning_PPO.py run.num_agents=10 run.should_tune=False run.recurrent=True model=recurrent ppo=recurrent +hyperparams/recurrent=config1"

eval ${cmd}
