"""
This example previews the results for trained agent so we can see the final policy
"""
from functools import partial

from ray import tune
from ray.tune import register_env

from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldOfficeLabelingFunctionWrapper, \
    OfficeWorldPlantLabelingFunctionWrapper, OfficeWorldCoffeeLabelingFunctionWrapper, \
    OfficeWorldMailLabelingFunctionWrapper
from rm_marl.new_stack.algos.algo import PPORM, PPORMConfig
from rm_marl.new_stack.env.multi_env_with_rm import make_multi_agent_with_rm
from rm_marl.new_stack.utils.env import env_creator, hydra_env_creator


# Open the agent from the path
results_path = "/home/rp218/ray_results/partial_andrew_coffee_mail_10_0.9979081153869629/PPORM_2024-10-27_15-36-42"

env_config = {
    "name": "gym_subgoal_automata:OfficeWorldCoffeeMail-v0",
    "render_mode": "human",
    "seed": 123,
    "label_factories": [
        partial(OfficeWorldOfficeLabelingFunctionWrapper, sensor_true_confidence=1, sensor_false_confidence=1),
        partial(OfficeWorldPlantLabelingFunctionWrapper, sensor_true_confidence=1, sensor_false_confidence=1),
        partial(OfficeWorldCoffeeLabelingFunctionWrapper, sensor_true_confidence=0.99790811538696291, sensor_false_confidence=0.99790811538696291),
        partial(OfficeWorldMailLabelingFunctionWrapper, sensor_true_confidence=1, sensor_false_confidence=1),
        # OfficeWorldALabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
        # OfficeWorldBLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
        # OfficeWorldCLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
        # OfficeWorldDLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
    ],
    "num_agents": 10,
}



# Load the env
register_env("env", make_multi_agent_with_rm(hydra_env_creator(env_config)))

tuner = tune.Tuner.restore(
    path=results_path,
    trainable=PPORM,
    # _resume_config=ResumeConfig(
    #     finished=ResumeConfig.ResumeType.RESUME,
    #     unfinished=ResumeConfig.ResumeType.RESUME,
    #     errored=ResumeConfig.ResumeType.RESUME,
    # )
)

# We assume there is only one result; we can modify this
results = tuner.get_results().get_best_result()
print(results)

config = PPORMConfig()
config.environment(
    env="env",
    env_config={
        "num_agents": env_config["num_agents"],
    },
)
config.evaluation(
    evaluation_config=PPORMConfig.overrides(
        entropy_coeff=0.0,
        explore=False,
        env_config={
            "seed": env_config["seed"],
        },
    ),
)
ppo = config.build()
ppo.restore(results.get_best_checkpoint())

ppo.evaluate()

print("done")
