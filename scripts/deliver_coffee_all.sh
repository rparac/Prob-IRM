#!/bin/bash
cd ..

seeds=(0 100 200 300 400)
noise_levels=(1 0.9979081153869629 0.995305061340332 0.9814815521240234)

directory="all_deliver_coffee"
for seed in "${seeds[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="${directory}_${seed}_${noise_level}"
    # run noise on all three
    python submit_rcs_script.py ${directory} ${name} \
      dqrm_coffee_world.py env/office-world@env=delier_coffee run=dqrm_coffee_world \
        +experiment=vanilla_all_symmetric_error x=${noise_level} \
        run.name=${directory}/${name} run.seed=${seed}
  done
done

# Running on login.hx1.hpc.ic.ac.uk
