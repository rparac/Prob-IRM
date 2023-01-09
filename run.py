import os

import hydra
from omegaconf import DictConfig, OmegaConf

from rm_marl.trainer import Trainer


def _instantiate(obj):
    obj = hydra.utils.instantiate(obj)
    obj = {k: v for k, v in obj.items() if not k.startswith("_")}
    return obj

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:

    run_config = cfg.run.copy()
    if not cfg.run.log_dir.startswith("/"):
        cfg.run.log_dir = os.path.join(os.path.dirname(__file__), cfg.run.log_dir)
    
    if cfg.run.training:
        env_mode = "local"
        agents = _instantiate(cfg.env.agents)
    else:
        env_mode = "shared"
        agents = Trainer.load(cfg.run.path).agents

    envs = _instantiate(cfg.env)[env_mode]
    
    trainer = Trainer(envs, agents)
    trainer.run(cfg.run)

if __name__ == "__main__":
    run()