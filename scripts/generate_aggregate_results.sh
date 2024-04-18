#!/bin/bash
cd ..

log_dir_reg=('/gpfs/home/rp218/rm-marl/saved_logs/coffee_error_*' '/gpfs/home/rp218/rm-marl/saved_logs/all_deliver_coffee_*')
output_dir=('/gpfs/home/rp218/rm-marl/saved_logs/coffee_error_' '/gpfs/home/rp218/rm-marl/saved_logs/all_deliver_coffee_')

for i in "${!log_dir_reg[@]}"; do
  python submit_rcs_script.py "generate_summaries" scripts/generate_summaries.py ${log_dir_reg[i]} ${output_dir[i]}
done
