"""
Quick script to run hydra with condor launcher
Run the file as
python submit_script.py <DIRECTORY> <NAME> ../dqrm_coffee_world.py <ARGS>

"""
import os
import shutil
import stat
import subprocess
import sys
import time
from typing import List

script_directory = "outputs"


pbs_script_gpu = """#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=200Gb:ngpus=1:gpu_type=A100

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

export PATH=$PATH:/gpfs/home/rp218/bin

module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

cd ${HOME}/rm-marl
conda activate rm_marl
"""

pbs_script_gpu2 = f"""#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=100Gb:ngpus=1

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

export PATH=$PATH:/gpfs/home/rp218/bin
export PYTHONPATH=$PYTHONPATH:/gpfs/home/rp218/rm-marl

module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0


cd $HOME/rm-marl
conda activate new
conda install -c git
pip install -r to_install.txt
"""


def get_pbs_script_base(experiment_directory: str):
    # return pbs_script_gpu2

    return f"""#!/bin/bash
    #PBS -l walltime=24:00:00
    #PBS -l select=1:ncpus=4:mem=200Gb

    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

    export PATH=$PATH:/gpfs/home/rp218/bin

    module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

    cd $HOME/rm-marl
    conda activate new
    """


def run_pbs(args, name, experiment_directory):
    python_run = f"python {' '.join(args)}"

    # generate scripts
    pbs_out = f"{script_directory}/{experiment_directory}/{name}.pbs"
    if not os.path.exists(f"{script_directory}/{experiment_directory}"):
        os.makedirs(f"{script_directory}/{experiment_directory}")

    with open(pbs_out, 'w') as f:
        f.write(get_pbs_script_base(experiment_directory))
        f.write('\n')
        f.write(python_run)

    result = subprocess.run(['qsub', '-q', 'hx', pbs_out], stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    print("Successfully ran all the scripts")


if __name__ == "__main__":
    arguments = sys.argv
    directory = arguments[1]
    name = arguments[2]
    args = arguments[3:]

    os.makedirs(script_directory, exist_ok=True)
    run_pbs(args, name, directory)
