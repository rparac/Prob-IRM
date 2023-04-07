import numpy as np
import pygame
import random

from .base_grid import BaseGridEnv
from .wrappers import LabelingFunctionWrapper

class CookieEnv(BaseGridEnv):

    def __init__(
        self, render_mode=None, size=6, positions=None, file=None
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
            img = font.render(label, True, self.COLOR_MAPPING["K"])
            canvas.blit(img, self.pix_square_size * (pos + 0.33))
        elif label[0] == "A":
            font = pygame.font.SysFont(None, 50)
            img = font.render(label, True, self.COLOR_MAPPING["K"])
            canvas.blit(img, self.pix_square_size * (pos + 0.25))

    def _get_obs(self):
        obs = {}
        if self.is_in_top_room(self.agent_locations["A1"]):
            for p, ls in self.positions.items():
                if self.is_in_top_room(p):
                    for l in ls:
                        obs[l] = np.array(p)
        elif self.is_in_right_room(self.agent_locations["A1"]):
            for p, ls in self.positions.items():
                if self.is_in_right_room(p):
                    for l in ls:
                        obs[l] = np.array(p)
        elif self.is_in_left_room(self.agent_locations["A1"]):
            for p, ls in self.positions.items():
                if self.is_in_left_room(p):
                    for l in ls:
                        obs[l] = np.array(p)
        elif self.is_in_hallway(self.agent_locations["A1"]):
            for p, ls in self.positions.items():
                if self.is_in_hallway(p):
                    for l in ls:
                        if not l.startswith("W"):
                            obs[l] = np.array(p)
        return obs

    def _get_info(self):
        return {}
    
    def is_in_top_room(self, pos):
        return pos[1] < 2 and pos[0] > 1 and pos[0] < 5
    
    def is_in_left_room(self, pos):
        return pos[0] < 2 and pos[1] > 3
    
    def is_in_right_room(self, pos):
        return pos[0] > 4 and pos[1] > 3
    
    def is_in_hallway(self, pos):
        return not any([
            self.is_in_top_room(pos), 
            self.is_in_left_room(pos), 
            self.is_in_right_room(pos)
        ])
    
    def step(self, actions):

        for aid, action in actions.items():
            direction = self._action_to_direction[self.Actions(action)]
            new_agent_pos = np.clip(
                self.agent_locations[aid] + direction, 0, self.size - 1
            )
            if self._can_enter(new_agent_pos):
                if tuple(self.agent_locations[aid]) != tuple(new_agent_pos):
                    self.positions[tuple(self.agent_locations[aid])].remove(aid)
                    self.positions[tuple(new_agent_pos)].append(aid)

                    if self._button_is_pressed():
                        self._randomize_goal()

        terminated = self._goal_is_reached()
        reward = 1 if terminated else 0  # Binary sparse rewards

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
        
    def _can_enter(self, new_pos):
        for _label, positions in self.postions_by_type("W").items():
            if tuple(new_pos) in positions:
                return False
        return True

    def _button_is_pressed(self):
        for pos in self.postions_by_type("B").values():
            if tuple(self.agent_locations["A1"]) in pos:
                return True
        return False

    def _goal_is_reached(self):
        for pos in self.postions_by_type("G").values():
            if tuple(self.agent_locations["A1"]) in pos:
                return True
        return False

    def _randomize_goal(self):
        gp = self.postions_by_type("G")
        if gp:
            for pos in gp["G"]:
                self.positions[pos].remove("G")

        if random.random() > 0.5:
            self.positions[(0,4)] = ["G"]
        else:
            self.positions[(6,4)] = ["G"]

class CookieLabelingFunctionWrapper(LabelingFunctionWrapper):
    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        agent_locations = obs or self.agent_locations
        prev_agent_locations = prev_obs or self.prev_agent_locations
        labels = []

        if self.is_in_top_room(agent_locations["A1"]):
            labels.append("t")
        elif self.is_in_left_room(agent_locations["A1"]):
            labels.append("l")
        elif self.is_in_right_room(agent_locations["A1"]):
            labels.append("r")

        b_positions = self.postions_by_type("B").get("BG", [])
        g_positions = self.postions_by_type("G").get("G", [])

        if self._agent_has_moved_to(
            "A1", prev_agent_locations, agent_locations, b_positions
        ):
            labels.append("b")

        if self._agent_has_moved_to(
            "A1", prev_agent_locations, agent_locations, g_positions
        ):
            labels.append("ce")
        elif g_positions and ((
            self.is_in_left_room(g_positions[0]) and self.is_in_left_room(agent_locations["A1"])
           ) or (
            self.is_in_right_room(g_positions[0]) and self.is_in_right_room(agent_locations["A1"])
           )):
            labels.append("cv")

        # return ["_".join(labels)] if labels else []
        return labels

    @staticmethod
    def _agent_has_entered_room(agent, from_loc, to_loc, room_func):
        return (
            not room_func(from_loc[agent])
            and room_func(to_loc[agent])
        )
    
    @staticmethod
    def _agent_has_exited_room(agent, from_loc, to_loc, room_func):
        return (
            not room_func(to_loc[agent])
            and room_func(from_loc[agent])
        )

    @staticmethod
    def _agent_has_moved_to(agent, from_loc, to_loc, positions):
        return (
            tuple(from_loc[agent]) not in positions
            and tuple(to_loc[agent]) in positions
        )