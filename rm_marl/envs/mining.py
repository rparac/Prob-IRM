"""
Implements the Mining environment introduced in the paper
Noisy Symbolic Abstractions for Deep RL: A case study with Reward Machines.

This is a simple single agent environment. The goal of the task is to dig gold and bring it
back to the depot(goal). The episode terminates when the agent reaches the depot. The agent
gets a positive reward only when it is back in the terminal state with gold
"""


from enum import IntEnum

import gymnasium
import numpy as np
import pygame

from rm_marl.envs.base_grid import BaseGridEnv
from rm_marl.envs.wrappers import LabelingFunctionWrapper


class MiningEnv(BaseGridEnv):
    class Actions(IntEnum):
        none = 0
        up = 1
        right = 2
        down = 3
        left = 4
        dig = 5

    def __init__(
            self, render_mode=None, size=6, positions=None, file=None, max_steps=200,
    ):
        super().__init__(render_mode, size, positions, file)

        self.action_space = gymnasium.spaces.Discrete(len(self.Actions))
        self._action_to_direction = {
            self.Actions.none: np.array([0, 0]),
            self.Actions.up: np.array([0, -1]),
            self.Actions.right: np.array([1, 0]),
            self.Actions.down: np.array([0, 1]),
            self.Actions.left: np.array([-1, 0]),
        }
        self.max_steps = max_steps

        self.dug_gold = False
        self.just_dug_gold = False
        self.tried_digging = False
        self.goal_reached = False

        self.num_steps = 0

        pos_max = (self.size, self.size)
        self.unflatten_obs_space = gymnasium.spaces.Dict({
            "tried_digging": gymnasium.spaces.Discrete(2),
            "dug_gold": gymnasium.spaces.Discrete(2),
            "goal_reached": gymnasium.spaces.Discrete(2),
            "pos": gymnasium.spaces.MultiDiscrete(list(pos_max), dtype=np.int32),
        })
        self.observation_space = gymnasium.spaces.Dict(
            {"A1": gymnasium.spaces.utils.flatten_space(self.unflatten_obs_space)})

    def _draw_component(self, label, pos, canvas):
        if label[0] == "W":
            pygame.draw.rect(
                canvas,
                self.COLOR_MAPPING[label[-1]],
                pygame.Rect(
                    self.pix_square_size * pos,
                    (self.pix_square_size, self.pix_square_size),
                ),
            )
        elif label[0] == "B":
            pygame.draw.circle(
                canvas,
                self.COLOR_MAPPING[label[-1]],
                (pos + 0.5) * self.pix_square_size,
                self.pix_square_size / 3,
            )
        elif label[0] == "G":
            pygame.draw.rect(
                canvas,
                self.COLOR_MAPPING[label[-1]],
                pygame.Rect(
                    self.pix_square_size * pos,
                    (self.pix_square_size, self.pix_square_size),
                ),
            )
        elif label[0] == "A":
            font = pygame.font.SysFont(None, 50)
            img = font.render(label, True, self.COLOR_MAPPING["K"])
            canvas.blit(img, self.pix_square_size * (pos + 0.25))

    def _get_obs(self):
        a1_pos = self.postions_by_type("A1")["A1"][0]
        obs_a1 = {"tried_digging": self.tried_digging, "dug_gold": self.just_dug_gold,
                  "goal_reached": self.goal_reached,
                  "pos": np.array(a1_pos)}
        # obs_a1 = {"dug_gold": self.just_dug_gold, "goal_reached": self.goal_reached, "pos": np.array(a1_pos)}
        obs = {"A1": gymnasium.spaces.utils.flatten(self.unflatten_obs_space, obs_a1)}
        return obs

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.dug_gold = False
        self.just_dug_gold = False
        self.goal_reached = False
        self.tried_digging = False

        self.num_steps = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, actions):
        self.just_dug_gold = False
        self.tried_digging = False
        for aid, action in actions.items():
            if self.Actions(action) == self.Actions.dig:
                self.tried_digging = True
                self.just_dug_gold = self._contains_gold(self.agent_locations[aid])
                self.dug_gold = self.dug_gold or self.just_dug_gold
            else:
                direction = self._action_to_direction[self.Actions(action)]
                new_agent_pos = np.clip(
                    self.agent_locations[aid] + direction, 0, self.size - 1
                )
                if self._can_enter(new_agent_pos):
                    if tuple(self.agent_locations[aid]) != tuple(new_agent_pos):
                        self.positions[tuple(self.agent_locations[aid])].remove(aid)
                        self.positions[tuple(new_agent_pos)].append(aid)

        self.goal_reached = self._goal_is_reached()
        terminated = self.goal_reached
        reward = 1 if terminated and self.dug_gold else -0.01  # Binary sparse rewards

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.num_steps += 1

        return observation, reward, terminated, self.num_steps >= self.max_steps, info

    def _can_enter(self, new_pos):
        for _label, positions in self.postions_by_type("W").items():
            if tuple(new_pos) in positions:
                return False
        return True

    def _goal_is_reached(self):
        for pos in self.postions_by_type("G").values():
            if tuple(self.agent_locations["A1"]) in pos:
                return True
        return False

    def _contains_gold(self, curr_pos):
        for _label, positions in self.postions_by_type("BY").items():
            if tuple(curr_pos) in positions:
                return True
        return False


class MiningLabelingFunctionWrapper(LabelingFunctionWrapper):
    # Constructor to ensure type safety
    def __init__(self, env: MiningEnv):
        super().__init__(env)
        self.env = env

    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        labels = []

        unwrapped_obs = gymnasium.spaces.unflatten(self.env.unflatten_obs_space, obs["A1"])
        if unwrapped_obs["dug_gold"]:
            labels.append('by')

        if unwrapped_obs["goal_reached"]:
            labels.append('g')

        return labels