from typing import Dict

import torch


def get_annealed_exploration_rate(num_steps, expl_init, expl_end, anneal_steps):
    return max((expl_end - expl_init) * num_steps / anneal_steps + expl_init, expl_end)


def embed_labels(labels: Dict[str, float]) -> torch.Tensor:
    x = [labels[key] for key in sorted(labels.keys())]
    return torch.tensor(x)
