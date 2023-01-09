from .buttons import ButtonsEnv
from gym.envs.registration import register

register(
    id='rm-marl/Buttons-v0',
    entry_point='rm_marl.envs:ButtonsEnv',
    max_episode_steps=300,
    disable_env_checker=True
)