from omegaconf import ListConfig
from ray import tune


# Converts from config to a function
def _to_tune(param):
    if param["tune_func"] == 'choice':
        # Convert ListConfig (possibly nested) to pure list
        options = [list(item) if isinstance(item, ListConfig) else item for item in param["options"]]
        return tune.choice(options)
    if param["tune_func"] == 'uniform':
        return tune.uniform(*param["options"])
    if param["tune_func"] == 'loguniform':
        return tune.loguniform(*param["options"])
    if param["tune_func"] == 'randint':
        return tune.randint(*param["options"])


def from_hydra_config(conf):
    new_config = {}
    for k, v in conf.items():
        # Ignore helper keys
        if k.startswith('_'):
            continue

        if v.best_value is not None:
            new_config[k] = v.best_value
        else:
            new_config[k] = _to_tune(v)

    return new_config
