#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor
# Need to change 3 things: recurrent=True, model=recurrent, hyperparmeters=recurrent/config5

cd ..

# TODO: seeds
# seeds=(0 100 200 300 400)
# num_agents=(10)
# use_rm_options=(False True)
ex_mult_options=(1) # 2 5 10 100)
noise_levels=(1) # 0.9979081153869629 0.995305061340332 0.9814815521240234)

nodes=1
ncpus=64
ram=128

directory="different_aut_deliver_coffee"
for ex_mult in "${ex_mult_options[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    #name="${directory}_${use_rm}_${noise_level}"
    run_subdirectory=${directory}_multiplier_${ex_mult}
    name=${run_subdirectory}_${noise_level}

    # run noise on all three
    # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} ray_tests/simple_test.py
    python submit_rcs_old_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
      ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.seed=123 \
        rm_learner.ex_penalty_multiplier=${ex_mult} \
        rm_learner.min_penalty=${ex_mult} \
        run.use_perfect_rm=False run.num_agents=1 run.should_tune=True \
	run.tune_config.num_samples=1 \
	run.num_env_runners=20 \
	run.wandb.key=680ad332869d9761ae2b6bdd70cdbc068674d47b \
	run.render_freq=20 \
        +hyperparams/with_rm=config5 \
        +experiment=vanilla_coffee_symmetric_error x=${noise_level}
    # python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
    # ray_tests/hydra_RM_learning_PPO.py run.name=${name} run.use_perfect_rm=True run.num_agents=${num_agent} run.should_tune=True run.num_env_runners=40 run.tune_config.num_samples=200
  done
done

	# run.wandb.project=${run_subdirectory} \
	# run.wandb.run_name=${noise_level} \ - with  just this it failed
# Running on login.hx1.hpc.ic.ac.uk
