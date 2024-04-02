#!/bin/bash

cd ..

noise_levels=(0.9 0.95 0.99 1)

for item in "${noise_levels[@]}"; do
  name="mail_error_coffee_mail_${item}"
  # run noise on all three
  python submit_condor_script.py ${name} dqrm_coffee_world.py env/office-world@env=coffee_mail run=dqrm_coffee_world +experiment=vanilla_coffee_symmetric_error x=${item} run.name=${name}
done
