#!/bin/bash

# Runs the visit abcd task for 5 seeds and noise levels with the noise on the a sensor

cd ..

seeds=(0 100 200 300 400)
use_rm_options=(True False)
noise_levels=(1 0.9989626407623291 0.997668981552124 0.9907407760620117)

nodes=1
ncpus=32
ram=256

directory="visit_abcd_a"
for seed in "${seeds[@]}"; do
  for use_rm in "${use_rm_options[@]}"; do
    for noise_level in "${noise_levels[@]}"; do
      #name="${directory}_${use_rm}_${noise_level}"
      run_subdirectory=${directory}_$([ "$use_rm" = True ] && echo "perfect_rm" || echo "rm_learning")
      name=${run_subdirectory}_${noise_level}_${seed}
  
      python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} 1 \
        ray_tests/hydra_RM_learning_PPO.py env/office-world@env=visit_abcd run.name=${name} \
          run.seed=${seed} \
          rm_learner.ex_penalty_multiplier=8 \
          rm_learner.min_penalty=8 \
          run.use_perfect_rm=${use_rm} run.num_agents=10 \
          run.num_env_runners=30 run.stop_iters=500 \
          +hyperparams/with_rm=configabcd \
          +experiment=vanilla_a_symmetric_error x=${noise_level} 
    done
  done
done
