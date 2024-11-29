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
    if param["tune_func"] == 'grid_search':
        return tune.grid_search(param["options"])
    raise ValueError(f"Unknown tune func {param['tune_func']}")


def from_hydra_config(conf):
    new_config = {}
    for k, v in conf.items():
        # Ignore helper keys
        if k.startswith('_'):
            continue

        if v['best_value'] is not None:
            new_config[k] = v['best_value']
        else:
            new_config[k] = _to_tune(v)

    return new_config


# Hacky solution. In the ideal world we could just set one value and use $ interpolation for the rest
# Hydra doesn't support overriding multiple values at once with optuna
#  We override values that should be overriden with this function
def manual_value_override(cfg):
    if 'manual_overrides' not in cfg:
        return

    override_values = cfg['manual_overrides']

    for override_value in override_values:
        if override_value in cfg["env"]["overridable"]:
            command = f"cfg.{override_value} = {cfg['x']}"
            exec(command)
