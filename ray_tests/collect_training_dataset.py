from typing import List
import torch
import pandas as pd
import gymnasium as gym

class LabelingFunctionDataset(torch.nn.Dataset):
    def __init__(self, ds_file: str, propositions: List[str]):
        self.data = pd.read_csv(ds_file)
        self._propositions = propositions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        obs = row["obs"]
        labels = row[all_observations]
        return obs, labels


env_config = {
    "name": "gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
    "render_mode": "rgb_array",
    "seed": 0,
    "max_episode_length": 300,
    "num_random_seeds": None,
}

env = gym.make(env_config["name"], render_mode=env_config["render_mode"],
                params={"generation": "random", "environment_seed": env_config["seed"],
                        "hide_state_variables": True, "num_plants": 1})

all_observations = env.unwrapped.get_observables()

data = {
    observable: []
    for observable in all_observations
}
data["obs"] = []

num_runs = 100
for i in range(num_runs):
    done = False
    obs, info = env.reset()
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        data["obs"].append(obs)
        labels_seen = info["observations"]
        for observable in all_observations:
            seen = observable in labels_seen
            data[observable].append(seen)


dataset_file = "training_dataset.csv"
df = pd.DataFrame(data)
df.to_csv(dataset_file, index=False)

