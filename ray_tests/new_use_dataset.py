import gymnasium as gym

from rm_marl.envs.pretrained_label_extractor import PretrainedLabelExtractor
from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter

env_config = {
    "name": "gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
    "render_mode": "rgb_array",
    "seed": 0,
    "max_episode_length": 100,
    "num_random_seeds": 1,
    "tb_storage_path": "runs/labeling_function_training"
}
curr_id = 0

env = gym.make(env_config["name"], render_mode=env_config["render_mode"],
                params={"generation": "random", "environment_seed": env_config["seed"] + curr_id,
                        "hide_state_variables": True, "num_plants": 1})
env = NewGymSubgoalAutomataAdapter(env, max_episode_length=env_config["max_episode_length"], num_random_seeds=env_config["num_random_seeds"])  # type: ignore

label_extractor = PretrainedLabelExtractor(env, env_config["tb_storage_path"])
obs, info = env.reset()
print(label_extractor.get_labels(obs, info))