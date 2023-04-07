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
        agents = _instantiate(cfg.env.agents)
        local_envs = _instantiate(cfg.env)["local"]
    else:
        agents = Trainer.load(cfg.run.path).agents
        local_envs = None
    
    shared_envs = _instantiate(cfg.env)["shared"]
    
    trainer = Trainer(local_envs, shared_envs, agents)
    # from copy import deepcopy
    # trainer = Trainer(deepcopy(shared_envs), shared_envs, agents)
    trainer.run(cfg.run)

if __name__ == "__main__":
    run()