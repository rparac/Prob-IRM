#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

seeds=(0) # 100 200 300 400)
use_rm_options=(True)
noise_levels=(1 0.9911642074584961)
# noise_levels=(0.9977762699127197 0.9961941242218018 0.9911642074584961)

nodes=2
ncpus=32
ram=250

directory="waterworld_rgnm_unrestricted"
for seed in "${seeds[@]}"; do
  for use_rm in "${use_rm_options[@]}"; do
    for noise_level in "${noise_levels[@]}"; do
      #name="${directory}_${use_rm}_${noise_level}"
      run_subdirectory=${directory}_$([ "$use_rm" = True ] && echo "perfect_rm" || echo "rm_learning")
      name=${run_subdirectory}-${noise_level}_${seed}
      y=$(printf "%.4f" "$noise_level")  # Convert the number to a string with 4 decimal places
      y=${y:2}                           # Extract substring from the third character onward
  
      # run noise on all three
      # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/simple_test.py
      python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
        ray_tests/hydra_RM_learning_PPO.py env/water-world@env=red_green_no_magenta run.name=${name} \
	  run.seed=${seed} \
	  env.use_restricted_observables=false \
          rm_learner.ex_penalty_multiplier=8 \
          rm_learner.min_penalty=4 \
          run.use_perfect_rm=${use_rm} run.num_agents=1 run.should_tune=True \
  	      run.tune_config.num_samples=1 \
          run.num_env_runners=30 run.stop_iters=40000 \
	  run.wandb.project=${run_subdirectory} \
          run.wandb.key=680ad332869d9761ae2b6bdd70cdbc068674d47b \
	  run.render_freq=1000 \
          +hyperparams/with_rm=configabcd \
          +experiment=vanilla_red_symmetric_error x=${noise_level} 
    done
  done
done

# Running on login.hx1.hpc.ic.ac.uk
