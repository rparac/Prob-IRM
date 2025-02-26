#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

seeds=(0 100 200 300 400)
# num_agents=(10)
# use_rm_options=(False True)
use_rm_options=(False) # (True) # False) 
noise_levels=(1 0.9979081153869629 0.995305061340332 0.9814815521240234)

nodes=1
ncpus=16
ram=128

directory="ppo_deliver_coffee"
for seed in "${seeds[@]}"; do
  for use_rm in "${use_rm_options[@]}"; do
    for noise_level in "${noise_levels[@]}"; do
      #name="${directory}_${use_rm}_${noise_level}"
      run_subdirectory=${directory}_$([ "$use_rm" = True ] && echo "perfect_rm" || echo "rm_learning")
      name=${run_subdirectory}_${noise_level}_${seed}
  
      # run noise on all three
      python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} 1 \
        ray_tests/hydra_RM_learning_PPO.py env/office-world@env=deliver_coffee run.name=${name} \
          run.seed=${seed} \
	  run.no_rm=True \
          run.num_agents=10 run.should_tune=True \
  	      run.tune_config.num_samples=1 \
          run.num_env_runners=15 run.stop_iters=500 \
	  run.render_freq=1000000 \
          +hyperparams/with_rm=configabcd \
          +experiment=vanilla_coffee_symmetric_error x=${noise_level} 
      # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
      # ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.use_perfect_rm=True run.num_agents=${num_agent} run.should_tune=True run.num_env_runners=40 run.tune_config.num_samples=200
    done
  done
done

	# run.wandb.project=${run_subdirectory} \
	# run.wandb.run_name=${noise_level} \ - with  just this it failed
# Running on login.hx1.hpc.ic.ac.uk
