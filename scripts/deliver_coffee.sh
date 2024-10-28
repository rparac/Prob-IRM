#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

# TODO: seeds
# seeds=(0 100 200 300 400)
num_agents=(3) # (10)
noise_levels=(1) # 0.9979081153869629 0.995305061340332 0.9814815521240234)

nodes=2
ncpus=64
ram=128

directory="less_partial_andrew_deliver_coffee"
for num_agent in "${num_agents[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="${directory}_${num_agent}_${noise_level}"
    # run noise on all three
    # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/simple_test.py
    python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
      ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.seed=127 \
        run.use_perfect_rm=True run.num_agents=${num_agent} run.should_tune=True \
	run.num_env_runners=20 run.tune_config.num_samples=4 \
	run.tune_config.scheduler.min_grace_period=100 \
        +hyperparams/with_rm=less_partial_andrew \
        +experiment=vanilla_coffee_symmetric_error x=${noise_level}
    # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
    # ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.use_perfect_rm=True run.num_agents=${num_agent} run.should_tune=True run.num_env_runners=40 run.tune_config.num_samples=200
  done
done

# Running on login.hx1.hpc.ic.ac.uk
