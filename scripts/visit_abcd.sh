#!/bin/bash
cd ..

seeds=(123 233 333 433 533)
noise_levels=(1 0.99 0.95 0.9)

for seed in "${seeds[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="visit_abcd_${seed}_${noise_level}"
    python submit_rcs_script.py ${name} dqrm_coffee_world.py env/office-world@env=visit_abcd run=visit_abcd +experiment=vanilla_a_symmetric_error x=${noise_level} run.name=${name} run.seed=${seed}
  done
done

# Running on login.hx1.hpc.ic.ac.uk
