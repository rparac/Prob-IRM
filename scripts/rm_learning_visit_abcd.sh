#!/bin/bash
cd ..

seeds=(0 100 200 300 400)
use_rm_options=(True False)
noise_levels=(1 0.9989626407623291 0.997668981552124 0.9907407760620117)

nodes=1
ncpus=32
ram=256

directory="long_visit_abcd_a"
for seed in "${seeds[@]}"; do
  for use_rm in "${use_rm_options[@]}"; do
    for noise_level in "${noise_levels[@]}"; do
      #name="${directory}_${seed}_${noise_level}"
      name=${directory}_$([ "$use_rm" = True ] && echo "perfect_rm" || echo "rm_learning")_${noise_level}_${seed}
      # run noise on all three
      python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
        ray_tests/hydra_RM_learning_PPO.py env/office-world@env=visit_abcd run.name=${name} \
          run.seed=${seed} \
          rm_learner.ex_penalty_multiplier=8 \
          rm_learner.min_penalty=8 \
          env.max_episode_length=1000 \
	  run.crash_iter=501 \
	  run.tune_config.checkpoint_freq=100 \
          run.use_perfect_rm=${use_rm} run.num_agents=10 run.should_tune=True \
  	      run.tune_config.num_samples=1 \
          run.num_env_runners=50 run.stop_iters=10000 \
	  run.render_freq=1000000 \
          +hyperparams/with_rm=configabcd \
          +experiment=vanilla_a_symmetric_error x=${noise_level} 
    done
  done
done

# Running on login.hx1.hpc.ic.ac.uk
# run.wandb.key=680ad332869d9761ae2b6bdd70cdbc068674d47b \
# run.render_freq=50 \
