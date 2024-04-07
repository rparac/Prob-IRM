#!/bin/bash

cd ..

noise_levels=(0.95 0.97 0.99 1)

for item in "${noise_levels[@]}"; do
  name="visit_abcd_a_error_${item}"
  # run noise on all three
  python submit_condor_script.py ${name} dqrm_coffee_world.py env/office-world@env=visit_abcd run=visit_abcd +experiment=vanilla_a_symmetric_error x=${item} run.name=${name}
done

# Running on batch1