#!/bin/bash
cd ..

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor
# Need to change 3 things: recurrent=True, model=recurrent, hyperparmeters=recurrent/config5

seeds=(0 100 200 300 400)
use_rm_options=(False) 
noise_levels=((1 0.9979081153869629 0.995305061340332 0.9814815521240234)

nodes=1
ncpus=32
ram=256

# directory="new_more_complex_after_fix_10_envs_long_visit_abcd_a"
directory="recurrent_coffee_mail"
for seed in "${seeds[@]}"; do
  for use_rm in "${use_rm_options[@]}"; do
    for noise_level in "${noise_levels[@]}"; do
      #name="${directory}_${seed}_${noise_level}"
      name=${directory}_$([ "$use_rm" = True ] && echo "perfect_rm" || echo "rm_learning")_${noise_level}_${seed}
      # run noise on all three
      python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
        ray_tests/hydra_RM_learning_PPO.py env/office-world@env=deliver_coffee_mail run.name=${name} \
          run.seed=${seed} \
	  run.recurrent=True \
	  run.tune_config.checkpoint_freq=100 \
          run.use_perfect_rm=${use_rm} run.num_agents=10 run.should_tune=True \
  	      run.tune_config.num_samples=1 \
          run.num_env_runners=30 run.stop_iters=500 \
	  run.render_freq=1000000 \
	  model=recurrent \
          +hyperparams/recurrent=configabcdshorter \
          +experiment=vanilla_coffee_symmetric_error x=${noise_level} 
    done
  done
done

# Running on login.hx1.hpc.ic.ac.uk
# run.wandb.key=680ad332869d9761ae2b6bdd70cdbc068674d47b \
# run.render_freq=50 \
