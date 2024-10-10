#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

nodes=4
ncpus=64
ram=128 # Gb

seeds=(0) # 100 200 300 400)
noise_levels=(1) # 0.9979081153869629 0.995305061340332 0.9814815521240234)
# wandb_name="single_agent_rm"
# wandb_name="single_agent_rm_learning"
# wandb_name="10_agents_rm"
wandb_name="10_agents_rm_tune"
# wandb_name="10_agents_rm_learning"
# wandb_name="single_agent_tune"
name="with_hyperparam_experiment"

cmd="python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/hydra_RM_learning_PPO.py run.num_agents=10 run.should_tune=False +hyperparams/with_rm=config1"

eval ${cmd}
