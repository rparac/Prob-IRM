from collections import Counter

import gymnasium as gym
from ray import tune
from ray.rllib.env import EnvContext
from ray.rllib.env.multi_agent_env import make_multi_agent


def make_env(env_id, idx, capture_video, run_name):
    def thunk(_env_ctx: EnvContext):
        counter = _env_ctx["counter"]
        curr_id = counter["id"]
        counter["id"] += 1
        if (capture_video or _env_ctx.get("capture_video")) and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array",
                           params={"generation": "random", "environment_seed": curr_id,
                                   "hide_state_variables": True})
            env = gym.wrappers.RecordVideo(env, f"videos/{_env_ctx.worker_index}/{run_name}")
        else:
            # env = gym.make("CartPole-v1")
            env = gym.make(env_id,
                           params={"generation": "random", "environment_seed": curr_id,
                                   "hide_state_variables": True})
        env = gym.wrappers.FlattenObservation(env)
        # env = gym.experimental.wrappers.DtypeObservationV0(env, **{"dtype": np.float32})
        # env = gym.wrappers.TransformObservation(
        #     env, **{"f": lambda obs: obs.astype("float32")}
        # )
        return env

    return thunk


env_cls = make_multi_agent(make_env('gym_subgoal_automata:OfficeWorldDeliverCoffee-v0', 0, False, "test"))
env = env_cls({"num_agents": 3, "counter": Counter()})
obs, info = env.reset()
print("Done")
