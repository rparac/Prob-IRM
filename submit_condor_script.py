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


condor_script_base = """#!/bin/bash

# Fail if there is an error
set -e
set -o pipefail

export TMPDIR=/tmp
export PATH=$PATH:/homes/rp218/bin


# Activate python environment
source /vol/rp218-tmp/miniconda3/etc/profile.d/conda.sh
conda activate rm_marl

"""

condor_cmd_base = """
#############################
##
## Condor job specification
##
##############################

universe        = vanilla

# This specifies what commandline arguments should be passed to the executable.
arguments       = $(Process)

# Only execute on lab machines due to ILASP requirement
requirements    = regexp("^(arc|beech|curve|edge|gpu|oak|point|ray|texel|vertex|willow)[0-9]{2}", TARGET.Machine)
"""


def generate_condor_script(args: List[str], name: str) -> str:
    python_run = f"python {' '.join(args)}"

    out_file = f"{script_directory}/script_{name}.sh"

    with open(out_file, 'w') as f:
        f.write(condor_script_base)
        f.write('\n')
        f.write(python_run)
        f.write('\n')

    # Make file executable
    st = os.stat(out_file)
    os.chmod(out_file, 0o755)
    return out_file


def run_condor(args, name):

    # generate scripts
    condor_cmd_out = f"{script_directory}/condor_full.cmd"
    with open(condor_cmd_out, 'w') as f:
        f.write(condor_cmd_base)
        f.write('\n')
        f.write(f'output          = script_{name}.$(Process).out\n')
        f.write(f'error           = script_{name}.$(Process).err\n')
        f.write(f'log             = script_{name}.log\n')
        shell_script = generate_condor_script(args, name)
        f.write(f'executable      = {shell_script}\n')
        f.write('\n')
        f.write(f'queue {1}\n')

    result = subprocess.run(['condor_submit', condor_cmd_out], stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    print("Successfully ran all the scripts")


if __name__ == "__main__":
    arguments = sys.argv
    name = arguments[1]
    args = arguments[2:]

    os.makedirs(script_directory, exist_ok=True)
    run_condor(args, name)

