"""
Quick script to run hydra with condor launcher
Run the file as
python submit_script.py <NAME> ray_tests/hydra_RM_learning_PPO.py <ARGS>

"""
import os
import shutil
import stat
import subprocess
import sys
from typing import List

script_directory = "outputs"


def pbs_script_base(nodes, ncpus, ram):
    return f"""#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select={nodes}:ncpus={ncpus}:mem={ram}Gb
cd $EPHEMERAL/rm-marl
export PYTHONPATH=$PYTHONPATH:/rds/general/user/rp218/ephemeral/rm-marl
export PATH=$PATH:/rds/general/user/rp218/home/bin
export RAY_RESULTS_DIR=$EPHEMERAL/ray_results
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate custom-ray

# Launch a ray cluster
ray start --head --node-ip-address=$(hostname -i) --port=6379
head_ip=$(hostname -i)
echo "Head node started on $head_ip"

# Wait a few seconds to ensure the head node is up
sleep 5
"""


def run_pbs(args, name, experiment_directory, nodes, ncpus, ram):
    python_run = f"python {' '.join(args)}"

    # generate scripts
    pbs_out = f"{script_directory}/{experiment_directory}/{name}.pbs"
    if not os.path.exists(f"{script_directory}/{experiment_directory}"):
        os.makedirs(f"{script_directory}/{experiment_directory}")

    with open(pbs_out, 'w') as f:
        f.write(pbs_script_base(nodes, ncpus, ram))
        f.write('\n')
        f.write(python_run)

    result = subprocess.run(['qsub', pbs_out], stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    print("Successfully ran all the scripts")


if __name__ == "__main__":
    arguments = sys.argv
    _nodes = int(arguments[1])
    _ncpus = int(arguments[2])
    _ram = int(arguments[3])
    directory = arguments[5]
    name = arguments[5]
    args = arguments[6:]

    os.makedirs(script_directory, exist_ok=True)
    run_pbs(args, name, directory, _nodes, _ncpus, _ram)
