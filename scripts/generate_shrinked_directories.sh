#!/bin/bash
log_dirs=("coffee_mail" "with_rm_coffee_mail" "all_deliver_coffee" "visit_abcd_a" "with_rm_visit_abcd_a")

directory="processing"
for log_dir in "${log_dirs[@]}"; do
  name="${directory}_${log_dir}"
  python submit_rcs_old_script.py 128 ${directory} ${name} \
    generate_small_tensorboard_logs.py ${log_dir}
done
