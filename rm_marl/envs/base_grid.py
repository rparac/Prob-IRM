import abc
import copy
from collections import defaultdict
from enum import Enum

import gym
import numpy as np
import pygame


class BaseGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    COLOR_MAPPING = {
        "K": (0, 0, 0),
        "Y": (255, 255, 102),
        "G": (178, 255, 102),
        "R": (255, 102, 102),
    }

    class Actions(Enum):
        up = 0  # move up
        right = 1  # move right
        down = 2  # move down
        left = 3  # move left
        none = 4  # none

    def __init__(self, render_mode=None, size=6, positions=None, file=None):
        if file is not None:
            size, positions = self.load_from_file(file)

        assert positions is not None, "`positions` must be specified"

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.original_positions = copy.deepcopy(positions)
        self.positions = defaultdict(list)
        for p, l in self.original_positions.items():
            self.positions[p].append(l)
        self.active_flags = defaultdict(self._active_flag_constructor)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.action_space = gym.spaces.Discrete(len(self.Actions))
        self._action_to_direction = {
            self.Actions.up: np.array([0, -1]),
            self.Actions.right: np.array([1, 0]),
            self.Actions.down: np.array([0, 1]),
            self.Actions.left: np.array([-1, 0]),
            self.Actions.none: np.array([0, 0]),
        }

        self.observation_space = gym.spaces.Dict(
            {
                a: gym.spaces.Box(0, size - 1, shape=(2,), dtype=int)
                for a in self.agent_locations.keys()
            }
        )

    @property
    def pix_square_size(self): 
        return self.window_size / self.size

    @property
    def agent_locations(self): 
        return {
            l: np.array(p) for p, ls in self.positions.items() for l in ls if l.startswith("A")
        }

    @staticmethod
    def _active_flag_constructor():
        return True

    @abc.abstractmethod
    def _get_obs(self):
        raise NotImplementedError("_get_obs")

    @abc.abstractmethod
    def _get_info(self):
        raise NotImplementedError("_get_info")

    @abc.abstractmethod
    def _draw_component(self, label, pos, canvas):
        raise NotImplementedError("_draw_component")

    def postions_by_type(self, type, pos=None):
        pos = pos or self.positions
        positions = defaultdict(list)
        for p, ls in pos.items():
            for l in ls:
                if l.startswith(type):
                    positions[l].append(p)
        return positions

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.active_flags.clear()
        self.positions.clear()
        for p, l in self.original_positions.items():
            self.positions[p].append(l)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    @abc.abstractmethod
    def step(self, actions):
        raise NotImplementedError("step")

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pygame.init()
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for pos, labels in self.positions.items():
            for label in labels:
                if not self.active_flags[label]:
                    continue
                pos = np.array(pos)
                self._draw_component(label, pos, canvas)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def load_from_file(file):
        with open(file, "r") as f:
            ls = [l.strip() for l in f.readlines()]

        ls = ls[1::2]

        num_cols = ls[0].count("|") - 1
        num_rows = len(ls)

        assert num_cols == num_rows, "The grid must be squared"

        positions = {}
        for r, row in enumerate(ls):
            row = row.split("|")[1:-1]
            for c, col in enumerate(row):
                if col.strip():
                    positions[(c, r)] = col.strip()

        return num_rows, positions
