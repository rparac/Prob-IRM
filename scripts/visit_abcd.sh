#!/bin/bash
cd ..

# TODO: seeds
# seeds=(0 100 200 300 400)
num_agents=(10)
noise_levels=(1 0.9989626407623291 0.997668981552124 0.9907407760620117)

nodes=2
ncpus=64
ram=128

directory="visit_abcd_a"
for num_agent in "${num_agents[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="${directory}_${seed}_${noise_level}"
    # run noise on all three
    python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
      ray_tests/hydra_RM_learning_PPO.py env/office-world@env=visit_abcd run.name=${name} \
        run.use_perfect_rm=True run.num_agents=${num_agent} run.should_tune=True \
        run.num_env_runners=20 \
        +hyperparams/with_rm=andrew \
        +experiment=vanilla_a_symmetric_error x=${noise_level} 
  done
done

# Running on login.hx1.hpc.ic.ac.uk
