# MARL with RM learning

This repo uses [hydra.cc](https://hydra.cc/) with the config defined in `config/`. It allows the user to override the config parameter directly via the command line.

Although the entrypoint of the code is in `run.py`, a lot of the objects are created on the fly according to the configuration specified. For a more transparent way of running the code, have a look in `run_manual.py`

## Training

```bash
python run.py env=buttons run.name=buttons-exp1 run.training=True
```

## Testing
```bash
python run.py env=buttons run.name=buttons-exp1-test run.training=False run.total_episodes=1 run.recording_freq=1 run.path=<PATH FROM TRAINING>
```

## Results

Results (including the reward, the loss, the number of steps and a video of the environment) can be seen in `Tensorboard` pointing to the `logs` directory.

```bash
tensorboard --logdir logs/
```

## Experiments

### RendezVous

```bash
ENV=rendezvous AGENT_TYPE=rm ./run_all.sh

ENV=rendezvous AGENT_TYPE=rm_learning ./run_all.sh

ENV=rendezvous AGENT_TYPE=onestate ./run_all.sh
```

### ThreeButtons

```bash
ENV=threebuttons AGENT_TYPE=rm ./run_all.sh

ENV=threebuttons AGENT_TYPE=rm_learning ./run_all.sh

ENV=threebuttons AGENT_TYPE=onestate ./run_all.sh
```