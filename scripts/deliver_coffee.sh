#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

# TODO: seeds
# seeds=(0 100 200 300 400)
# num_agents=(10)
# use_rm_options=(False True)
use_rm_options=(True) # False) # True)
noise_levels=(1) # 0.9979081153869629 0.995305061340332 0.9814815521240234)

nodes=2
ncpus=64
ram=128

directory="deliver_coffee"
for use_rm in "${use_rm_options[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    #name="${directory}_${use_rm}_${noise_level}"
    name=${directory}_$([ "$use_rm" = True ] && echo "perfect_rm" || echo "rm_learning")_${noise_level}

    # run noise on all three
    # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/simple_test.py
    python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
      ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.seed=123 \
        run.use_perfect_rm=${use_rm} run.num_agents=10 run.should_tune=True \
	run.tune_config.num_samples=30 \
	run.num_env_runners=20 \
        +hyperparams/with_rm=stability_tuning \
        +experiment=vanilla_coffee_symmetric_error x=${noise_level}
    # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
    # ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.use_perfect_rm=True run.num_agents=${num_agent} run.should_tune=True run.num_env_runners=40 run.tune_config.num_samples=200
  done
done

# Running on login.hx1.hpc.ic.ac.uk
