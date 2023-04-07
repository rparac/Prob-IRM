from typing import TYPE_CHECKING

import numpy as np
import pygame

from .base_grid import BaseGridEnv
from .wrappers import LabelingFunctionWrapper, RandomLabelingFunctionWrapper

if TYPE_CHECKING:
    from ..reward_machine import RewardMachine

_DEFAULT_POSITIONS = {
    (0, 1): "YB",
    (0, 2): "KW",
    (0, 4): "KW",
    (1, 2): "KW",
    (1, 3): "YW",
    (1, 4): "KW",
    (1, 5): "GW",
    (2, 2): "KW",
    (2, 4): "KW",
    (3, 2): "KW",
    (3, 3): "GB",
    (3, 5): "RB",
    (4, 2): "KW",
    (4, 3): "KW",
    (4, 4): "KW",
    (4, 5): "KW",
    (5, 3): "RW",
    (5, 5): "GL",
}


class ButtonsEnv(BaseGridEnv):
    def __init__(
        self, render_mode=None, size=6, positions=_DEFAULT_POSITIONS, file=None
    ):
        super().__init__(render_mode, size, positions, file)

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
            font = pygame.font.SysFont(None, 50)
            img = font.render("G", True, self.COLOR_MAPPING["K"])
            canvas.blit(img, self.pix_square_size * (pos + 0.25))
        elif label[0] == "A":
            font = pygame.font.SysFont(None, 50)
            img = font.render(label, True, self.COLOR_MAPPING["K"])
            canvas.blit(img, self.pix_square_size * (pos + 0.25))

    def _get_obs(self):
        return self.agent_locations.copy()

    def _get_info(self):
        return {}

    def step(self, actions):

        for aid, action in actions.items():
            direction = self._action_to_direction[self.Actions(action)]
            new_agent_pos = np.clip(
                self.agent_locations[aid] + direction, 0, self.size - 1
            )
            if self._can_enter(new_agent_pos):
                self.positions[tuple(self.agent_locations[aid])].remove(aid)
                self.positions[tuple(new_agent_pos)].append(aid)

        self._check_open_walls()

        terminated = self._goal_is_reached()
        reward = 1 if terminated else 0  # Binary sparse rewards

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _goal_is_reached(self):
        for agent_loc in self.agent_locations.values():
            for pos in self.postions_by_type("G").values():
                if tuple(agent_loc) in pos:
                    return True
        return False

    def _can_enter(self, new_pos):
        for label, positions in self.postions_by_type("W").items():
            if self.active_flags[label] and tuple(new_pos) in positions:
                return False
        return True

    def _check_open_walls(self):
        by_positions = self.postions_by_type("B").get("BY", [])
        bg_positions = self.postions_by_type("B").get("BG", [])
        br_positions = self.postions_by_type("B").get("BR", [])

        if any(tuple(loc) in by_positions for loc in self.agent_locations.values()):
            self.open_walls("Y")
        if any(tuple(loc) in bg_positions for loc in self.agent_locations.values()):
            self.open_walls("G")
        if (
            sum([tuple(loc) in br_positions for loc in self.agent_locations.values()])
            > 1
        ):
            self.open_walls("R")

    def open_walls(self, color):
        self.active_flags[f"W{color.upper()}"] = False

    @staticmethod
    def open_walls_Y(e):
        e.open_walls("Y")

    @staticmethod
    def open_walls_G(e):
        e.open_walls("G")

    @staticmethod
    def open_walls_R(e):
        e.open_walls("R")


class ButtonsLabelingFunctionWrapper(LabelingFunctionWrapper):
    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        agent_locations = obs or self.agent_locations
        prev_agent_locations = prev_obs or self.prev_agent_locations
        labels = []

        by_positions = self.postions_by_type("B").get("BY", [])
        bg_positions = self.postions_by_type("B").get("BG", [])
        br_positions = self.postions_by_type("B").get("BR", [])
        g_positions = self.postions_by_type("G").get("Gl", [])

        if self._agent_has_moved_to(
            "A1", prev_agent_locations, agent_locations, by_positions
        ):
            labels.append("by")

        if self._agent_has_moved_to(
            "A2", prev_agent_locations, agent_locations, bg_positions
        ):
            labels.append("bg")

        if self._agent_has_moved_to(
            "A2", prev_agent_locations, agent_locations, br_positions
        ):
            labels.append("a2br")
        elif self._agent_has_moved_to(
            "A2", agent_locations, prev_agent_locations, br_positions
        ):
            labels.append("a2lr")

        if self._agent_has_moved_to(
            "A3", prev_agent_locations, agent_locations, br_positions
        ):
            labels.append("a3br")
        elif self._agent_has_moved_to(
            "A3", agent_locations, prev_agent_locations, br_positions
        ):
            labels.append("a3lr")

        if (
            tuple(prev_agent_locations["A2"]) in br_positions
            and tuple(agent_locations["A2"]) in br_positions
            and tuple(prev_agent_locations["A3"]) in br_positions
            and tuple(agent_locations["A3"]) in br_positions
        ):
            labels.append("br")

        if self._agent_has_moved_to(
            "A1", prev_agent_locations, agent_locations, g_positions
        ):
            labels.append("g")

        return labels

    @staticmethod
    def _agent_has_moved_to(agent, from_loc, to_loc, positions):
        return (
            tuple(from_loc[agent]) not in positions
            and tuple(to_loc[agent]) in positions
        )


class ButtonsRandomLabelingFunctionWrapper(RandomLabelingFunctionWrapper):
    @staticmethod
    def can_open_wall_R_A2_A3(e):
        events = [l for l in e.flatten_trace if l in ("bg", "a2br", "a2lr", "a3br", "a3lr")]
        return events and "bg" in events and any(l == events[-1] for l in ("a2br", "a3br"))

    @staticmethod
    def can_open_wall_G_A3(e):
        return e.active_flags["WG"]

    @staticmethod
    def can_open_wall_Y_A2(e):
        return e.active_flags["WY"]

    @staticmethod
    def can_open_wall_R_A1(e):
        return e.active_flags["WR"] and "by" in e.flatten_trace
