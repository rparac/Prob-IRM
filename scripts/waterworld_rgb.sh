#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

seeds=(0 100 200)
use_rm_options=(True False)
noise_levels=(1 0.9990105628967285 0.9977762699127197 0.9911642074584961)

nodes=1
ncpus=32
ram=256

directory="waterworld_rgb_unrestricted"
# directory="waterworld_rgb_unrestricted_min_memory"
for seed in "${seeds[@]}"; do
  for use_rm in "${use_rm_options[@]}"; do
    for noise_level in "${noise_levels[@]}"; do
      #name="${directory}_${use_rm}_${noise_level}"
      run_subdirectory=${directory}_$([ "$use_rm" = True ] && echo "perfect_rm" || echo "rm_learning")
      name=${run_subdirectory}-${noise_level}_${seed}
  
      # run noise on all three
      # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/simple_test.py
      python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} 20 \
        ray_tests/hydra_RM_learning_PPO.py env/water-world@env=red_green_blue run.name=${name} \
	  run.seed=${seed} \
	  env.use_restricted_observables=false \
          rm_learner.ex_penalty_multiplier=8 \
          rm_learner.min_penalty=4 \
	  rm_learner.replay_experience=false \
	  rm_learner.max_container_size=null \
          run.use_perfect_rm=${use_rm} run.num_agents=1 run.should_tune=True \
  	      run.tune_config.num_samples=1 \
          run.num_env_runners=20 run.stop_iters=50000  \
	  run.continue_training=true \
	  run.tune_config.checkpoint_freq=2500 \
	  run.crash_iter=2500 \
          +hyperparams/with_rm=configabcd \
          +experiment=vanilla_red_symmetric_error x=${noise_level} 
    done
  done
done

# Running on login.hx1.hpc.ic.ac.uk
