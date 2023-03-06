import numpy as np
import pygame

from .base_grid import BaseGridEnv
from .wrappers import LabelingFunctionWrapper, RandomLabelingFunctionWrapper

_DEFAULT_POSITIONS = {
    (0, 0): "G1",
    (5, 5): "G2",
    (3, 3): "RV",
}

class RendezVousEnv(BaseGridEnv):
    def __init__(
        self, render_mode=None, size=6, positions=_DEFAULT_POSITIONS, file=None
    ):
        super().__init__(render_mode, size, positions, file)
        self.rdv_satisfied = False

    def _draw_component(self, label, pos, canvas):
        if label[0] == "R":
            pygame.draw.circle(
                canvas,
                self.COLOR_MAPPING["G"],
                (pos + 0.5) * self.pix_square_size,
                self.pix_square_size / 3,
            )
        elif label[0] == "G":
            font = pygame.font.SysFont(None, 50)
            img = font.render(label, True, self.COLOR_MAPPING["K"])
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
            self.positions[tuple(self.agent_locations[aid])].remove(aid)
            self.positions[tuple(new_agent_pos)].append(aid)

        self._check_rdv()

        terminated = self._goal_is_reached()
        reward = 1 if terminated else 0  # Binary sparse rewards

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _check_rdv(self):
        if not self.rdv_satisfied:
            rdv_positions = self.postions_by_type("R")["RV"]
            self.rdv_satisfied = all(tuple(al) in rdv_positions for al in self.agent_locations.values())

    @staticmethod
    def satisfy_rdv(e):
        e.rdv_satisfied = True

    def _goal_is_reached(self):
        if not self.rdv_satisfied:
            return False
        goal_positions = self.postions_by_type("G")
        for agent_id, agent_loc in self.agent_locations.items():
            gp = goal_positions.get(f"G{agent_id[-1]}")
            if tuple(agent_loc) not in gp:
                return False
        return True

class RendezVousLabelingFunctionWrapper(LabelingFunctionWrapper):
    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        agent_locations = obs or self.agent_locations
        prev_agent_locations = prev_obs or self.prev_agent_locations
        labels = []

        rdv_positions = self.postions_by_type("R")["RV"]
        goal_positions = self.postions_by_type("G")

        if self._agent_has_moved_to(
            "A1", prev_agent_locations, agent_locations, rdv_positions
        ):
            labels.append("r1")
        elif self._agent_has_moved_to(
            "A1", agent_locations, prev_agent_locations, rdv_positions
        ):
            labels.append("nr1")

        if self._agent_has_moved_to(
            "A2", prev_agent_locations, agent_locations, rdv_positions
        ):
            labels.append("r2")
        elif self._agent_has_moved_to(
            "A2", agent_locations, prev_agent_locations, rdv_positions
        ):
            labels.append("nr2")

        if (
            tuple(prev_agent_locations["A1"]) in rdv_positions
            and tuple(agent_locations["A1"]) in rdv_positions
            and tuple(prev_agent_locations["A2"]) in rdv_positions
            and tuple(agent_locations["A2"]) in rdv_positions
        ):
            labels.append("r")

        if self._agent_has_moved_to(
            "A1", prev_agent_locations, agent_locations, goal_positions["G1"]
        ):
            labels.append("g1")

        if self._agent_has_moved_to(
            "A2", prev_agent_locations, agent_locations, goal_positions["G2"]
        ):
            labels.append("g2")

        return labels

    @staticmethod
    def _agent_has_moved_to(agent, from_loc, to_loc, positions):
        return (
            tuple(from_loc[agent]) not in positions
            and tuple(to_loc[agent]) in positions
        )


class RendezVousRandomLabelingFunctionWrapper(RandomLabelingFunctionWrapper):
    @staticmethod
    def can_satisfy_rdv_A1(e):
        events = [l for l in e.flatten_trace if l in ("r1",)]
        return events and events[-1] == "r1"

    @staticmethod
    def can_satisfy_rdv_A2(e):
        events = [l for l in e.flatten_trace if l in ("r2",)]
        return events and events[-1] == "r2"
