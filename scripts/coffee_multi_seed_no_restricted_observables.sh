#!/bin/bash
cd ..

seeds=(123 233 333 433 533)
noise_levels=(1 0.99 0.95 0.9)

for seed in "${seeds[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="coffee_error_${seed}_${noise_level}"
    # run noise on all three
    python submit_rcs_script.py ${name} dqrm_coffee_world.py env/office-world@env=deliver_coffee run=dqrm_coffee_world +experiment=vanilla_coffee_symmetric_error x=${noise_level} run.name=${name} run.seed=${seed} env.use_restricted_observables=false
  done
done

# Running on login.hx1.hpc.ic.ac.uk
