"""
Quick script to run hydra with condor launcher
Run the file as
python submit_script.py <DIRECTORY> <NAME> ../dqrm_coffee_world.py <ARGS>

Environment variables require special care on HX1
  - https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/applications/guides/pytorch/

The following is the content of env_vars in the custom-ray environment
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib64/graphviz:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/new/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/gpfs/home/rp218/bin
export RAY_RESULTS_DIR=/gpfs/home/rp218/ray_results
export PYTHONPATH=$PYTHONPATH:$HOME/rm-marl
export PYTHONPATH=$PYTHONPATH:$HOME/miniconda3/envs/custom-ray/lib/python3.10/site-packages
export PYTHONPATH=$PYTHONPATH:$HOME/rm-marl/src

"""
import os
import shutil
import socket
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

def get_pbs_script_base(nodes, ncpus, ram):
    # return pbs_script_gpu2

    return f"""#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select={nodes}:ncpus={ncpus}:mem={ram}Gb

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# export PATH=$PATH:/gpfs/home/rp218/bin
# export RAY_RESULTS_DIR=/gpfs/home/rp218/ray_results
# export PYTHONPATH=$PYTHONPATH:$HOME/rm-marl
# 
# export LD_LIBRARY_PATH=$HOME/bin:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$HOME/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$HOME/lib64/graphviz:$LD_LIBRARY_PATH

# Hacky solution for linking libpython3.10 - not sure why conda doesn't do it automatically
# export LD_LIBRARY_PATH=$HOME/miniconda3/envs/new/lib:$LD_LIBRARY_PATH


cd $HOME/rm-marl
echo "Run Home successfully" >&2
conda activate custom-ray
echo "Activated environment" >&2
# conda uninstall pip
# echo "Uninstalled pip" >&2
# conda install pip==21.2.4
# echo "Installed pip" >&2
conda install -c conda-forge git
echo "Installed git" >&2
# pip install scikit-learn==1.4.2
# pip install torch==2.3.1
pip install -r to_install.txt
echo "Installed deps" >&2
pip freeze

# Launch a ray cluster
echo "Starting cluster" >&2
# ray start --head --node-ip-address=192.0.0.1 --port 6379 --dashboard-agent-grpc-port=52364
head_ip=$(hostname -i)
echo "Cluster started on $head_ip" >&2
echo "Head node started on $head_ip"

# Wait a few seconds to ensure the head node is up
sleep 10
"""


def run_pbs(args, name, experiment_directory, nodes, ncpus, ram):
    python_run = f"python {' '.join(args)}"

    # generate scripts
    pbs_out = f"{script_directory}/{experiment_directory}/{name}.pbs"
    if not os.path.exists(f"{script_directory}/{experiment_directory}"):
        os.makedirs(f"{script_directory}/{experiment_directory}")

    with open(pbs_out, 'w') as f:
        f.write(get_pbs_script_base(nodes, ncpus, ram))
        f.write('\n')
        f.write(python_run)

    hostname = socket.gethostname()

    if "hx1" in hostname:
        cmd = ['qsub', '-q', 'hx', pbs_out]
    else:
        cmd = ['qsub', pbs_out]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
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
