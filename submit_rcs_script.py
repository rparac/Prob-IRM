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


pbs_script_gpu = """#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=18:mem=200Gb:ngpus=1:gpu_type=A100

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

export PATH=$PATH:/gpfs/home/rp218/bin

module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

cd ${HOME}/rm-marl
conda activate rm_marl
"""

pbs_script_base = """#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=18:mem=200Gb

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

export PATH=$PATH:/gpfs/home/rp218/bin

module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

cd ${HOME}/rm-marl
conda activate rm_marl
"""




def run_pbs(args, name):
    python_run = f"python {' '.join(args)}"

    # generate scripts
    pbs_out = f"{script_directory}/script_{name}.pbs"
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

