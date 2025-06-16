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
import time
from typing import List

# Directory where temporary SLURM scripts are dumped
script_directory = "outputs"


# SLURM job specification that is shared among processes
def get_slurm_base_script(experiment_directory: str):
    return f"""#!/bin/bash
    #SBATCH --time=24:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=4
    #SBATCH --mem=200G

    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

    export PATH=$PATH:$HOME/bin

    cd $HOME/rm-marl
    conda activate rm_marl
    """


def run_slurm(args, name, experiment_directory):
    # get the python command with all the arguments
    python_run = f"python {' '.join(args)}"

    # generate scripts
    slurm_out = f"{script_directory}/{experiment_directory}/{name}.sh"
    if not os.path.exists(f"{script_directory}/{experiment_directory}"):
        os.makedirs(f"{script_directory}/{experiment_directory}")

    # Generate SLURM script
    with open(slurm_out, 'w') as f:
        f.write(get_slurm_base_script(experiment_directory))
        f.write('\n')
        f.write(python_run)

    # Submit a script
    result = subprocess.run(['sbatch', slurm_out], stdout=subprocess.PIPE, text=True)
    print(result.stdout)


if __name__ == "__main__":
    arguments = sys.argv
    directory = arguments[1]
    name = arguments[2]
    args = arguments[3:]

    os.makedirs(script_directory, exist_ok=True)
    run_slurm(args, name, directory)
