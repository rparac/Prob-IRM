#!/bin/bash

cd ..

noise_levels=(0.95 0.97 0.99 1)

for item in "${noise_levels[@]}"; do
  name="symmetric_error_coffee_mail_${item}"
  # run noise on all three
  python submit_condor_script.py ${name} dqrm_coffee_world.py env/office-world@env=deliver_coffee_mail run=dqrm_coffee_world +experiment=vanilla_all_symmetric_error x=${item} run.name=${name}
done

# Running on batch2