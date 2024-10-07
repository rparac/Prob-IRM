#!/bin/bash

# Runs the deliver coffee task for 5 seeds and noise levels with the noise on the coffee sensor

cd ..

seeds=(0) # 100 200 300 400)
noise_levels=(1) # 0.9979081153869629 0.995305061340332 0.9814815521240234)
# wandb_name="single_agent_rm"
# wandb_name="single_agent_rm_learning"
# wandb_name="10_agents_rm"
# wandb_name="10_agents_rm_learning"
wandb_name="single_agent_tune"

directory="deliver_coffee"
for seed in "${seeds[@]}"; do
  for noise_level in "${noise_levels[@]}"; do
    name="${wandb_name}"

    cmd="python submit_rcs_script.py ${directory} ${name} ray_tests/RM_learning_PPO.py --enable-new-api-stack --stop-iters 10 --custom-num-agents 1"
    if [ "$wandb_name" = "single_agent_rm" ]; then
      cmd="$cmd --use-perfect-rm --wandb-run-name=single_agent_rm"
    elif [ "$wandb_name" = "single_agent_rm_learning" ]; then
      cmd="$cmd --wandb-run-name=single_agent_rm_learning"
    elif [ "$wandb_name" = "10_agents_rm" ]; then
      cmd="$cmd --custom-num-agents 10 --use-perfect-rm --wandb-run-name=10_agents_rm"
    elif [ "$wandb_name" = "10_agents_rm_learning" ]; then
      cmd="$cmd --custom-num-agents 10 --wandb-run-name=10_agents_rm_learning"
    elif [ "$wandb_name" = "single_agent_tune" ]; then
      cmd="$cmd --use-perfect-rm --num-samples 5" # --wandb-run-name=single_agent_tune"
    else
      cmd=cmd
    fi

#    cmd="$cmd --wandb-key=7aa07ea83aacbc6521c091eb60561b8637a7f512"
    eval ${cmd}
  done
done

