#!/bin/bash
cd ..

base_dir="/rds/general/user/rp218/ephemeral/logs"
# base_dirs=("$base_dir/all_coffee_mail" "$base_dir/all_deliver_coffee" "$base_dir/coffee_mail" "$base_dir/deliver_coffee" "$base_dir/visit_abcd_a" "$base_dir/with_rm_visit_abcd_a" "$base_dir/with_rm_visit_abcd_all")
base_dirs=("$base_dir/coffee_mail")

directory="generate_summaries"
for base_dir in "${base_dirs[@]}"; do
  name="generate_summaries"
  output="summary"
  python submit_rcs_old_script.py ${name} ${directory} \
    scripts/generate_summaries.py ${base_dir} ${output}
done
