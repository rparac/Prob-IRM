# MARL with RM learning

## Training

```bash
python run.py env=buttons run.name=buttons-exp1 run.training=True
```

## Testing
```bash
python run.py env=buttons run.name=buttons-exp1-test run.training=False run.total_episodes=1 run.recording_freq=1 run.path=<PATH FROM TRAINING>
```