from typing import Dict

import torch


def get_annealed_exploration_rate(num_steps, expl_init, expl_end, anneal_steps):
    return max((expl_end - expl_init) * num_steps / anneal_steps + expl_init, expl_end)


def embed_labels(labels: Dict[str, float] | None, num_observables: int) -> torch.Tensor:
    if labels is None:
        return torch.zeros(num_observables)

    x = [labels[key] for key in sorted(labels.keys())]
    return torch.tensor(x)
