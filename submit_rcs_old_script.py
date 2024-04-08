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


pbs_script_base = """#!/bin/bash
#PBS -l select=1:ncpus=8:mem=100Gb
#PBS -l walltime=48:00:00

module load tools/prod
module load anaconda3/personal

export PATH=$PATH:/rds/general/user/rp218/home/bin

source activate rm-marl
cd ${HOME}/rm-marl/
"""




def run_pbs(args, name):
    python_run = f"python {' '.join(args)}"

    # generate scripts
    pbs_out = f"{script_directory}/{name}.pbs"
    with open(pbs_out, 'w') as f:
        f.write(pbs_script_base)
        f.write('\n')
        f.write(python_run)

    result = subprocess.run(['qsub', '-q', 'hx', pbs_out], stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    print("Successfully ran all the scripts")


if __name__ == "__main__":
    arguments = sys.argv
    name = arguments[1]
    args = arguments[2:]

    os.makedirs(script_directory, exist_ok=True)
    run_pbs(args, name)

