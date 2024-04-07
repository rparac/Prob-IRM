#!/bin/bash
cd ..

seeds=(123 124 125 126 127)
noise_levels=(1)

for seed in "${seeds[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="visit_abcd_${seed}_${noise_level}"
    # run noise on all three
    python submit_rcs_script.py ${name} dqrm_coffee_world.py env/office-world@env=visit_abcd run=visit_abcd +experiment=vanilla_a_symmetric_error x=${item} run.name=${name} run.seed=${seed}

  done
done

# Running on login.hx1.hpc.ic.ac.uk
