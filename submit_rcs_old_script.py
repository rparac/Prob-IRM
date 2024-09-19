"""
Quick script to run hydra with condor launcher
Run the file as
python submit_script.py <NAME> ../dqrm_coffee_world.py <ARGS>

"""
import os
import shutil
import stat
import subprocess
import sys
from typing import List

script_directory = "outputs"


def pbs_script_base(ram):
    return f"""#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem={ram}Gb
cd $EPHEMERAL/rm-marl
export PYTHONPATH=$PYTHONPATH:/rds/general/user/rp218/ephemeral/rm-marl
export PATH=$PATH:/rds/general/user/rp218/home/bin
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ray
"""


def run_pbs(args, name, experiment_directory, ram):
    python_run = f"python {' '.join(args)}"

    # generate scripts
    pbs_out = f"{script_directory}/{experiment_directory}/{name}.pbs"
    if not os.path.exists(f"{script_directory}/{experiment_directory}"):
        os.makedirs(f"{script_directory}/{experiment_directory}")

    with open(pbs_out, 'w') as f:
        f.write(pbs_script_base(ram))
        f.write('\n')
        f.write(python_run)

    result = subprocess.run(['qsub', pbs_out], stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    print("Successfully ran all the scripts")


if __name__ == "__main__":
    arguments = sys.argv
    directory = arguments[1]
    name = arguments[2]
    args = arguments[3:]

    os.makedirs(script_directory, exist_ok=True)
    run_pbs(args, name, directory, ram=50)
