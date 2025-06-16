#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor
# Need to change 3 things: recurrent=True, model=recurrent, hyperparmeters=recurrent/config5

cd ..

# Manully increased max penalty becuase of 100
# Ran with higher max penalty because of prevoius experiment (should not make an impact)

seeds=(0 100 200 300 400)
threshold_options=(0 0.5 1 1.5 2)
noise_levels=(1 0.9979081153869629 0.995305061340332 0.9814815521240234)

nodes=1
ncpus=32
ram=256

directory="cross_entorpy_threshold"
for seed in "${seeds[@]}"; do
  for threshold in "${threshold_options[@]}"; do
    for noise_level in "${noise_levels[@]}"; do
      #name="${directory}_${use_rm}_${noise_level}"
      name=${directory}-${threshold}_${noise_level}_${seed}

      # run noise on all three
      python submit_rcs_old_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
        ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.seed=${seed} \
	  rm_learner.cross_entropy_threshold=${threshold} \
          run.use_perfect_rm=False run.num_agents=10 run.should_tune=True \
          run.tune_config.num_samples=1 \
          run.num_env_runners=30 \
          +hyperparams/with_rm=configabcd \
          +experiment=vanilla_coffee_symmetric_error x=${noise_level}
    done
  done
done

    # run.wandb.key=680ad332869d9761ae2b6bdd70cdbc068674d47b \
    # run.render_freq=20 \
	# run.wandb.project=${run_subdirectory} \
	# run.wandb.run_name=${noise_level} \ - with  just this it failed
# Running on login.hx1.hpc.ic.ac.uk
