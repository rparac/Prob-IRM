from typing import Optional, Union, Type

import numpy as np
from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext
from ray.rllib.utils import override

import gymnasium as gym
from ray.rllib.utils.typing import EnvCreator

from rm_marl.reward_machine import RewardMachine


def make_multi_agent_with_rm(
        env_name_or_creator: Union[str, EnvCreator],
) -> Type["MultiAgentEnv"]:
    class MultiEnv(MultiAgentEnv):
        def __init__(self, config: EnvContext = None):
            MultiAgentEnv.__init__(self)

            # Note(jungong) : explicitly check for None here, because config
            # can have an empty dict but meaningful data fields (worker_index,
            # vector_index) etc.
            # TODO(jungong) : clean this up, so we are not mixing up dict fields
            # with data fields.
            if config is None:
                config = {}
            num = config.pop("num_agents", 1)
            self._num_agents = num
            if isinstance(env_name_or_creator, str):
                self.envs = [gym.make(env_name_or_creator) for _ in range(num)]
            else:
                self.envs = []
                for i in range(num):
                    config["curr_id"] = i
                    self.envs.append(env_name_or_creator(config))
                # self.envs = [env_name_or_creator(config) for _ in range(num)]
            self.terminateds = set()
            self.truncateds = set()
            self.observation_spaces = self._build_observation_space()
            self.action_spaces = gym.spaces.Dict(
                {i: self.envs[i].action_space for i in range(num)}
            )
            self.agents = list(range(num))
            self.possible_agents = self.agents.copy()

        def _build_observation_space(self):
            return gym.spaces.Dict(
                {i: self.envs[i].observation_space for i in range(self._num_agents)}
            )

        def update_rm(self, rm: RewardMachine):
            for env in self.envs:
                env.update_rm(rm)

            self.observation_spaces = self._build_observation_space()

        @override(MultiAgentEnv)
        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            self.terminateds = set()
            self.truncateds = set()
            obs, infos = {}, {}
            for i, env in enumerate(self.envs):
                obs[i], infos[i] = env.reset(seed=seed, options=options)

            return obs, infos

        @override(MultiAgentEnv)
        def step(self, action_dict):
            obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}

            # The environment is expecting an action for at least one agent.
            if len(action_dict) == 0:
                raise ValueError(
                    "The environment is expecting an action for at least one agent."
                )

            for i, action in action_dict.items():
                obs[i], rew[i], terminated[i], truncated[i], info[i] = self.envs[
                    i
                ].step(action)
                if terminated[i]:
                    self.terminateds.add(i)
                if truncated[i]:
                    self.truncateds.add(i)
            # TODO: Flaw in our MultiAgentEnv API wrt. new gymnasium: Need to return
            #  an additional episode_done bool that covers cases where all agents are
            #  either terminated or truncated, but not all are truncated and not all are
            #  terminated. We can then get rid of the aweful `__all__` special keys!
            terminated["__all__"] = len(self.terminateds) + len(self.truncateds) == len(
                self.envs
            )
            truncated["__all__"] = len(self.truncateds) == len(self.envs)
            return obs, rew, terminated, truncated, info

        @override(MultiAgentEnv)
        def render(self):
            # This render method simply renders all n underlying individual single-agent
            # envs and concatenates their images (on top of each other if the returned
            # images have dims where [width] > [height], otherwise next to each other).
            render_images = [e.render() for e in self.envs]
            return render_images
            # if render_images[0].shape[1] > render_images[0].shape[0]:
            #     concat_dim = 0
            # else:
            #     concat_dim = 1
            # return np.concatenate(render_images, axis=concat_dim)

    return MultiEnv
