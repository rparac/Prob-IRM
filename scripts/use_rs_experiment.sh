#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

seeds=(0 100 200 300 400)
use_rs_options=(False True)
noise_levels=(1 0.9989626407623291 0.997668981552124 0.9907407760620117)

nodes=1
ncpus=32
ram=256

directory="rs_deliver_coffee"
for use_rs in "${use_rs_options[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    for seed in "${seeds[@]}"; do

      #name="${directory}_${use_rm}_${noise_level}"
      run_subdirectory=${directory}_$([ "$use_rs" = True ] && echo "with_shaping" || echo "no_shaping")
      name=${run_subdirectory}_${noise_level}_${seed}

      # run noise on all three
      # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/simple_test.py
      python submit_rcs_old_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
        ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.seed=${seed} \
          run.use_perfect_rm=False run.num_agents=10 run.should_tune=True \
	  run.use_rs=${use_rs} \
          run.tune_config.num_samples=1 \
  	  run.num_env_runners=30 \
          run.render_freq=10000000 \
          +hyperparams/with_rm=configabcd \
          +experiment=vanilla_all_symmetric_error x=${noise_level}
      # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
      # ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.use_perfect_rm=True run.num_agents=${num_agent} run.should_tune=True run.num_env_runners=40 run.tune_config.num_samples=200
    done
  done
done
# run.wandb.key=680ad332869d9761ae2b6bdd70cdbc068674d47b \
# Running on cx3

