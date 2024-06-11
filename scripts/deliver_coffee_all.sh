#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on all sensors

cd ..

seeds=(0 100 200 300 400)
noise_levels=(1 0.9979081153869629 0.995305061340332 0.9814815521240234)

directory="all_deliver_coffee"
for seed in "${seeds[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="${directory}_${seed}_${noise_level}"
    # run noise on all three
    python submit_rcs_script.py ${directory} ${name} \
      dqrm_coffee_world.py env/office-world@env=deliver_coffee run=dqrm_coffee_world \
        +experiment=vanilla_all_symmetric_error x=${noise_level} \
        run.name=${directory}/${name} run.seed=${seed}
  done
done

# Running on login.hx1.hpc.ic.ac.uk
