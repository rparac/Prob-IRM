#!/bin/bash
cd ..

# TODO: seeds
# seeds=(0 100 200 300 400)
# num_agents=(10)
use_rm_options=(False True)
noise_levels=(1 0.9989626407623291 0.997668981552124 0.9907407760620117)

nodes=2
ncpus=64
ram=128

directory="visit_abcd_a"
for use_rm in "${use_rm_options[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    #name="${directory}_${seed}_${noise_level}"
    name=${directory}_$([ "$use_rm" = True ] && echo "perfect_rm" || echo "rm_learning")_${noise_level}
    # run noise on all three
    python submit_rcs_script.py ${nodes} ${ncpus} ${ram} ${directory} ${name} \
      ray_tests/hydra_RM_learning_PPO.py env/office-world@env=visit_abcd run.name=${name} \
        run.use_perfect_rm=${use_rm} run.num_agents=10 run.should_tune=True \
        run.num_env_runners=20 run.stop_iters=1000 \
        +hyperparams/with_rm=less_partial_andrew \
        +experiment=vanilla_a_symmetric_error x=${noise_level} 
  done
done

# Running on login.hx1.hpc.ic.ac.uk
