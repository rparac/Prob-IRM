#!/bin/bash
cd ..

seeds=(123)
thresholds=(0.4 0.5 0.6 0.8 1 1.2)

for seed in "${seeds[@]}"; do
  for threshold in "${thresholds[@]}"; do
    name="cross_entropy_${threshold}"
    # run noise on all three
    python submit_rcs_script.py ${name} dqrm_coffee_world.py env/office-world@env=deliver_coffee_mail run=dqrm_coffee_world +experiment=vanilla_coffee_symmetric_error x=0.9 run.name=${name} run.seed=${seed} run.rm_learner_kws.cross_entropy_threshold=${threshold}
  done
done

# Running on login.hx1.hpc.ic.ac.uk
