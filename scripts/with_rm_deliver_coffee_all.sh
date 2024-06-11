#!/bin/bash
cd ..

seeds=(0 100 200 300 400)
noise_levels=(1 0.9989626407623291 0.997668981552124 0.9907407760620117)

directory="with_rm_deliver_coffee_all"
for seed in "${seeds[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="${directory}_${seed}_${noise_level}"
    # run noise on all three
    python submit_rcs_script.py ${directory} ${name} \
      with_automata.py env/office-world@env=deliver_coffee_mail run=dqrm_coffee_world \
        +experiment=vanilla_coffee_symmetric_error x=${noise_level} \
        run.name=${directory}/${name} run.seed=${seed}
  done
done

# Running on login.hx1.hpc.ic.ac.uk
