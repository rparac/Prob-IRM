#!/bin/bash

python RM_learning_PPO.py --enable-new-api-stack --wandb-project=prob-irm --stop-iters=25   \
  --use-perfect-rm --wandb-run-name=single_agent_rm \
  --wandb-key=INSERT