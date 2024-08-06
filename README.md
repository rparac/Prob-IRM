# Running an experiment on SLURM

## Setting up a conda environment

Create a conda environment with the required python version

```
conda create -n rm_marl python=3.10.12
conda activate rm_marl
pip install -r to_install.txt
```

My script assumes that this environment will be inside the job

## Installing ILASP and dot

The program relies on ILASP (for automata learning) and dot (for pdf generation).
I think the support is too slow, so I do everything on my own.
This is how I bypass the superuser requirements:

### $HOME/bin directory
The $HOME/bin directory contains the binaries that I need on my PATH, i.e. ILASP and dot.

```
# in .bashrc
export PATH=$HOME/bin:$PATH
```

### $HOME/lib64 folder

Both of these binaries are linking to dynamic dependencies.
These are contained in the `lib64` folder.

```
# in .bashrc
export LD_LIBRARY_PATH=$HOME/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib64/graphviz:$LD_LIBRARY_PATH
```

### Installing ILASP

ILASP binaries can be found here:
https://github.com/ilaspltd/ILASP-releases/releases

There is one for ubuntu.
If you have RHEL/Centos, I will send it over since this has not been published anywhere.

The program assumes ILASP binary is copied into the $HOME/bin directory (which I add to the path inside a script).

Its only dependency is libpython3.10 (if I recall correctly), which should be in $PATH because of conda.



### Installing dot

`dot` is used for generating automaton pdfs.

The `dot` on Ubuntu is installed with:
```
sudo apt install graphviz
```
However, sudo is mostly unavailable on clusters.
So, I 
1. Create a local docker container with the same OS as the server
2. Run the above command inside the container
3. Copy over the binary (into the `bin/` on the server) and all the installed libraries (into `lib64/` on the server)

#### Step 1
```
FROM redhat/ubi8-minimal:8.6
CMD ["sleep", "infinity"]
```
Here the Dockerfile I have been using. Change the image to whatever the correct distribution.

Run inside the directory with Dockerfile:
`docker build -t installation .`

Run the container:
`docker run installation`

In a new terminal, get its id (under CONTAINER_ID tab)
`docker ps`

Connect to it:
`docker exec -it <container_id> /bin/bash`

You can now install compabile binaries.

#### Step 2
Run 
`apt update && apt install graphviz`

#### Step 3
Get the required files:
Run:
`whereis dot` 
to fit its location.

Run:
`ldd <path_to_dot>` 
to get its depedencies.
These need to be copied over.

This is an example output on rhel:
```
[root@4390bfe09d2b lib64]# ldd /usr/bin/dot
	linux-vdso.so.1 (0x00007ffc6e2fb000)
	libgvc.so.6 => /lib64/libgvc.so.6 (0x000074f201774000)
	libltdl.so.7 => /lib64/libltdl.so.7 (0x000074f20156a000)
	libxdot.so.4 => /lib64/libxdot.so.4 (0x000074f201364000)
	libcgraph.so.6 => /lib64/libcgraph.so.6 (0x000074f20114c000)
	libpathplan.so.4 => /lib64/libpathplan.so.4 (0x000074f200f43000)
	libexpat.so.1 => /lib64/libexpat.so.1 (0x000074f200d07000)
	libz.so.1 => /lib64/libz.so.1 (0x000074f200aef000)
	libm.so.6 => /lib64/libm.so.6 (0x000074f20076d000)
	libcdt.so.5 => /lib64/libcdt.so.5 (0x000074f200566000)
	libc.so.6 => /lib64/libc.so.6 (0x000074f2001a1000)
	libdl.so.2 => /lib64/libdl.so.2 (0x000074f1fff9d000)
	/lib64/ld-linux-x86-64.so.2 (0x000074f201c1a000)
```

These files are symbolic links. We need to copy the actual binary (this will be e.g. `libgvc.so.6.1` - I think it can always be found with `libgvc.so.6.1.*`).

Copy all these files to the your machine (outside docker) - I usually make a simple bash script to do so. 
`e.g. docker cp <container_id>:/lib64/libgvc.so.6.1.*`
Don't forget the actual binary.
`e.g. docker cp <container_id>:/usr/bin/dot`

Copy the files to the server
```
scp <dependency_name> servername:~/lib64/
scp dot servername:~/bin/
```


Finally, we need to recreate the short names on the server - I also usually make a script for this.
Need to run for each dependency (inside lib64):
`ln -s libgvc.so.6 libgvc.so.6.*`


## Running pre-setup experiments

The `scripts` directory contains the scripts for the current experiments

`cd scripts/`

See the script that you may want to run, e.g. deliver_coffee.

Replace the `submit_rcs_script.py` (or `submit_rcs_old_script.py`) with `submit_slurm_script.py`.

Run:
`./deliver_coffee.sh`
This will launch a SLURM job for every combination of 5 possible seeds and 4 noise levels.

To enable checkpointing, please look at `visit_abcd_all.sh`. The two added parameters are `run.checkpoint_freq=1000 run.restart_from_checkpoint=True`
The default behaviour continues the most recent run.


## Modifying the existing experiment

The main script that I mostly run is `dqrm_coffee_world.py`, which starts the learning process.
It uses Hydra for configuration management. This allows us to quickly modify the running parameters as command line arguments.
For example, including when running the script
`run.checkpoint_freq=1000`
changes the checkpointing frequency to 1000 episodes.

To obtain parameters that can be modified, run:
`python dqrm_coffee_world.py --cfg job`

A particularly interesting one may be 
`run.rm_learner_kws.cross_entropy_threshold=$VALUE`
because of the recent bug

# Debugging

The `submit_slurm_script.py` is commented. Please look into it if there are any mistakes that I have made.