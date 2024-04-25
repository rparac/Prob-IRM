#!/bin/bash
cd ..

seeds=(123)
thresholds=(0.4 0.5 0.6 0.8 1 1.2)
noise_levels=(1 0.99 0.95 0.9)

directory="cross_entropy"
for seed in "${seeds[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    for threshold in "${thresholds[@]}"; do
      name="${directory}_${seed}_${noise_level}_${threshold}"
      python submit_rcs_script.py ${directory} ${name} \
       dqrm_coffee_world.py env/office-world@env=deliver_coffee run=dqrm_coffee_world \
        +experiment=vanilla_coffee_symmetric_error x=${noise_level} run.name=${name} run.seed=${seed} run.rm_learner_kws.cross_entropy_threshold=${threshold}
    done
      name="${directory}_${seed}_${noise_level}_none"
      python submit_rcs_script.py ${directory} ${name} \
       dqrm_coffee_world.py env/office-world@env=deliver_coffee run=dqrm_coffee_world \
        +experiment=vanilla_coffee_symmetric_error x=${noise_level} run.name=${name} run.seed=${seed} run.rm_learner_kws.use_cross_entropy=False

  done
done

# Running on login.hx1.hpc.ic.ac.uk
