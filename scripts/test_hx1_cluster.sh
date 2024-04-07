#!/bin/bash
cd ..

noise_levels=(0.99 1)

for item in "${noise_levels[@]}"; do
  name="test_${item}"
  # run noise on all three
  python submit_rcs_script.py ${name} dqrm_coffee_world.py env/office-world@env=deliver_coffee_mail run=debugging_coffee_mail +experiment=vanilla_coffee_symmetric_error x=${item} run.name=${name}
done

# Running on login.hx1.hpc.ic.ac.uk
