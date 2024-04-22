#!/bin/bash
cd ..

base_dirs=('/gpfs/home/rp218/rm-marl/saved_logs/')

directory="generate_summaries"
for base_dir in "${base_dirs[@]}"; do
  name="generate_summaries"
  python submit_rcs_script.py ${name} ${directory} \
    script/generate_summaries.py ${base_dir}
done
