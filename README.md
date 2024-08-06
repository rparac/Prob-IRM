# Prob-IRM

This is the source code for the KR2024 paper "Learning Robust Reward Machines from Noisy Labels".
Please reach out to the main author, Roko Parać (rp218@ic.ac.uk), for any questions regarding this source code or the paper itself.


## Dependencies


### Setting up a conda environment

Create a conda environment with the required python version

```
conda create -n rm_marl python=3.10.12
conda activate prob-irm
pip install -r requirements.txt
```

### Installing ILASP

ILASP is used for automata learning. You can install it by adding its binary to the system PATH. Binaries can be found here:
https://github.com/ilaspltd/ILASP-releases/releases.
The experiments have been run with v4.4.0

We recommend downloading the binary to $HOME/bin for consistency with this source code.
This directory can be added to path with the following command:
```export PATH="$HOME/bin:$PATH"```


### Installing dot

`dot` is used for generating automaton pdfs.

The `dot` on Ubuntu is installed with:
```
sudo apt install graphviz
```
The experiments have been run with version 2.43.0, although a wide variety of versions should probably work since its only purpose was to generate pdfs.


## Running pre-defined experiments

The `scripts` directory contains the scripts for running all the experiments demonstrated in the paper.
These have been run on Imperial's RCS (Research Compute Service) cluster. 
It is setup as a PBS cluster, but we have also attempted to use Condor and SLURM.

To run the standard Deliver Coffee experiment:
```
1. cd scripts/
2. ./deliver_coffee.sh
```

## Runnning custom experiments

The file `dqrm_coffee_world.py` is the starting point of our program.
It is set up using Hydra configuration management tool (https://hydra.cc/).
To see all the available configuration options run:
```
python dqrm_coffee_world.py --cfg job
```
These can be overriden from the command line, for example to change the number of episodes to 4000:
```
python dqrm_coffee_world.py run.total_episodes=4000
```

Please reach out if you are struggling to run an experiment.