# TODO: acknowledge Italian people

"""
Propositions (to implement):
 - pickaxe (p)
 - gem (g)
 - door (d)
 - lava (l)
"""

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import torch
import torchvision
from PIL import Image
from gymnasium import spaces
from gymnasium.spaces import Box

from rm_marl.envs.visual_minecraft.finite_state_machine import MooreMachine


resize = torchvision.transforms.Resize((64, 64))
normalize = torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    resize,
])


# tutta la griglia
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, formula, render_mode="human", state_type="symbolic", use_dfa_state=True, train=True, size=4, random_start=False):
        self.dictionary_symbols = ['P', 'L', 'D', 'G', 'E']
        vis_minecraft_folder = Path(__file__).parent.absolute()
        dir_prefix = f"{vis_minecraft_folder}/imgs"
        self._PICKAXE = f"{dir_prefix}/pickaxe.png"
        self._GEM = f"{dir_prefix}/gem.png"
        self._DOOR = f"{dir_prefix}/door.png"
        self._ROBOT = f"{dir_prefix}/robot.png"
        self._LAVA = f"{dir_prefix}/lava.jpg"

        self._train = train
        self.use_dfa_state = use_dfa_state
        self.max_num_steps = 50
        self.curr_step = 0
        self.random_start = random_start

        self.state_type = state_type
        self.size = size  # 4x4 world
        self.window_size = 512  # size of the window

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.formula = formula
        self.automaton = MooreMachine(self.formula[0], self.formula[1], self.formula[2],
                                      dictionary_symbols=self.dictionary_symbols,
                                      reward="sparse")

        self.set_for_dict = set(self.automaton.rewards)
        self.list_rew = sorted(self.set_for_dict)
        self.rew_dictionary = {}
        for idx, reward in enumerate(self.list_rew):
            self.rew_dictionary[reward] = idx

        self.task = self.formula[2]

        self.action_space = spaces.Discrete(4)
        if state_type == "symbolic":
            self.state_space_size = 2
        elif state_type == "image":
            self.state_space_size = (3, 64, 64)
        # 0 = GO_DOWN
        # 1 = GO_RIGHT
        # 2 = GO_UP
        # 3 = GO_LEFT
        self.input_size = 16 + self.automaton.num_of_states
        self._action_to_direction = {
            0: np.array([0, 1]),  # DOWN
            1: np.array([1, 0]),  # RIGHT
            2: np.array([0, -1]),  # UP
            3: np.array([-1, 0]),  # LEFT
        }

        self._gem_location = np.array([0, 3])
        self._pickaxe_location = np.array([1, 1])
        self._exit_location = np.array([3, 0])
        self._lava_location = np.array([3, 3])

        self._gem_display = True
        self._pickaxe_display = True
        self._robot_display = False if self._train else True

        self.rng = np.random.default_rng()

        self._starting_locations = []

        if state_type == "image":
            self.image_locations = {}
            for r in range(size):
                for c in range(size):
                    self._agent_location = np.array([r, c])
                    if render_mode == "human":
                        self._render_frame()
                        obss = self._get_obs(1)
                    else:
                        obss = self._render_frame()

                    obss = torch.tensor(obss.copy(), dtype=torch.float64) / 255
                    obss = torch.permute(obss, (2, 0, 1))
                    obss = resize(obss)
                    obss = normalize(obss)
                    obss = obss.cpu().numpy()
                    self.image_locations[r, c] = obss
            # normalization
            all_images = list(self.image_locations.values())
            # self._save_images(all_images)
            # raise RuntimeError("Done")
            all_img_tens = np.stack(all_images)
            self.stdev = np.std(all_img_tens, axis=0)
            self.mean = np.mean(all_img_tens, axis=0)

            # for r in range(size):
            #     for c in range(size):
            #         norm_img = (self.image_locations[r, c] - self.mean) / (self.stdev + 1e-5)
            #         self.image_locations[r, c] = norm_img

            # for k in self.image_locations.keys():
            #     self.image_locations[k] = np.transpose(self.image_locations[k], (1, 2, 0))

            self.observation_space = Box(
                low=-10.0, # loose values to account for standardized values
                high=10.0,
                shape=self.state_space_size,
                dtype=np.float64,
            )
        elif state_type == "symbolic":
            self.observation_space = spaces.MultiDiscrete([size, size])

    def _save_images(self, all_images):
        location = "dataset/visual_minecraft_new"
        os.makedirs(location, exist_ok=True)

        for i, img in enumerate(all_images):
            f_name = f"img_{i}"

            # np_img = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            # pil_img = Image.fromarray(np_img)

            pil_img = Image.fromarray(img)
            pil_img.save(f"{location}/{f_name}.png")

    def reset(self, seed=None, options=None):
        '''
        TUTTO IL RESET
        '''

        super().reset(seed=seed, options=options)

        self.curr_automaton_state = 0
        self.curr_step = 0

        if self.random_start:
            if seed is not None:
                self.rng = np.random.default_rng(seed)

            start_pos = self.rng.integers(0, self.size, size=2)
            forbidden_positions = np.stack([self._pickaxe_location, self._gem_location, self._exit_location, self._lava_location])
            while np.any(np.all(forbidden_positions == start_pos, axis=1)):
                start_pos = self.rng.integers(0, self.size, size=2)
            self._agent_location = start_pos
        else:
            self._agent_location = np.array([0, 0])

        self._starting_locations.append(self._agent_location)

        if self.render_mode == "human":
            self._render_frame()
        if self.state_type == "symbolic":
            if self.use_dfa_state:
                observation = np.array(list(self._agent_location) + [self.curr_automaton_state])
            else:
                observation = np.array(list(self._agent_location))
        elif self.state_type == "image":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                # print("one_hot_dfa_state: ", one_hot_dfa_state)
                observation = [np.array(one_hot_dfa_state), self.image_locations[
                    self._agent_location[0], self._agent_location[1]]]  # 1 FULL Img, 0 Just the square the robot is in
            else:
                observation = self.image_locations[self._agent_location[0], self._agent_location[1]]
        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))

        reward = 0
        info = self.rew_dictionary[reward]
        label_info = self._get_label_info()
        info = {"rew_dict": info}
        info.update(label_info)

        return observation, info

    def _current_symbol(self):
        if (self._agent_location == self._exit_location).all():
            return 2
        if (self._agent_location == self._pickaxe_location).all():
            return 0
        if (self._agent_location == self._gem_location).all():
            return 3
        if (self._agent_location == self._lava_location).all():
            return 1
        return 4

    def step(self, action):

        reward = -1
        self.curr_step += 1
        done = False

        # MOVEMENT
        if action == 0:
            direction = np.array([0, 1])
        elif action == 1:
            direction = np.array([1, 0])
        elif action == 2:
            direction = np.array([0, -1])
        elif action == 3:
            direction = np.array([-1, 0])

        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        sym = self._current_symbol()
        # print("symbol:", sym)
        self.new_automaton_state = self.automaton.transitions[self.curr_automaton_state][sym]
        # print("state:", self.curr_automaton_state)
        # print(self.automaton.acceptance)

        # if self.automaton.acceptance[self.curr_automaton_state]:
        #    reward = 100
        #    done = True
        if self.new_automaton_state == self.curr_automaton_state:
            reward = 0
        else:
            reward = self.automaton.rewards[self.new_automaton_state] - self.automaton.rewards[
                self.curr_automaton_state]
        potential = self.automaton.rewards[self.new_automaton_state]
        self.curr_automaton_state = self.new_automaton_state

        if self.render_mode == "human":
            self._render_frame()

        if self.state_type == "symbolic":
            if self.use_dfa_state:
                observation = np.array(list(self._agent_location) + [self.curr_automaton_state])
            else:
                observation = np.array(list(self._agent_location))
        elif self.state_type == "image":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                # print("one_hot_dfa_state: ", one_hot_dfa_state)
                observation = [np.array(one_hot_dfa_state),
                               self.image_locations[self._agent_location[0], self._agent_location[1]]]
            else:
                observation = self.image_locations[self._agent_location[0], self._agent_location[1]]

        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))

        #          success            failure                  timeout
        terminated = self.automaton.is_accepting_state(self.curr_automaton_state)
        truncated = (self.curr_step >= self.max_num_steps) and not terminated

        info = self._get_info(potential)
        info = {"rew_dict": info}
        label_info = self._get_label_info()
        info.update(label_info)

        return observation, reward, terminated, truncated, info  # , sym

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def get_automaton_specs(self):
        num_of_states = self.automaton.num_of_states
        num_of_symbols = len(self.dictionary_symbols)
        num_outputs = len(self.list_rew)
        transition_function = self.automaton.transitions
        automaton_rewards = [self.rew_dictionary[rew] for rew in self.automaton.rewards]
        return num_of_states, num_of_symbols, num_outputs, transition_function, automaton_rewards

    def _get_obs(self, full=1):
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )
        # Not sure why they invert the channels
        # img = img[:, :, ::-1]
        obs = None
        if full == 1:
            obs = img
        else:
            pix_square_size = (self.window_size / self.size)
            pix_square_size = int(pix_square_size)
            x = self._agent_location[0]
            y = self._agent_location[1]
            obs = img[int(y * pix_square_size):int((y + 1) * pix_square_size),
                  int(x * pix_square_size):int((x + 1) * pix_square_size)]
        return obs

    def _get_label_info(self):

        outcome = 0
        has_pickaxe = (self._agent_location == self._pickaxe_location).all()
        has_gem = (self._agent_location == self._gem_location).all()
        has_exit = (self._agent_location == self._exit_location).all()
        has_lava = (self._agent_location == self._lava_location).all()

        if has_pickaxe:
            outcome = 1
        elif has_gem:
            outcome = 2
        elif has_lava:
            outcome = 3
        elif has_exit:
            outcome = 4
        
        return {"label": outcome}

    def _get_info(self, reward):

        info = self.rew_dictionary[reward]
        return info

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size / self.size)

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        pickaxe = pygame.image.load(self._PICKAXE)
        gem = pygame.image.load(self._GEM)
        door = pygame.image.load(self._DOOR)
        robot = pygame.image.load(self._ROBOT)
        lava = pygame.image.load(self._LAVA)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())

            if self._pickaxe_display:
                self.window.blit(pickaxe, (
                    pix_square_size * self._pickaxe_location[0], pix_square_size * self._pickaxe_location[1]))
            if self._gem_display:
                self.window.blit(gem, (
                    pix_square_size * self._gem_location[0], 32 + pix_square_size * self._gem_location[1]))
            self.window.blit(door, (pix_square_size * self._exit_location[0], pix_square_size * self._exit_location[1]))
            self.window.blit(lava, (
                pix_square_size * self._lava_location[0] + 2, pix_square_size * self._lava_location[1] + 2))

            if self._robot_display:
                self.window.blit(robot,
                                 (pix_square_size * self._agent_location[0], pix_square_size * self._agent_location[1]))

            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            if self._pickaxe_display:
                canvas.blit(pickaxe, (
                    pix_square_size * self._pickaxe_location[0], pix_square_size * self._pickaxe_location[1]))
            if self._gem_display:
                canvas.blit(gem, (
                    pix_square_size * self._gem_location[0], 32 + pix_square_size * self._gem_location[1]))
            canvas.blit(door, (pix_square_size * self._exit_location[0], pix_square_size * self._exit_location[1]))
            canvas.blit(lava, (
                pix_square_size * self._lava_location[0] + 2, pix_square_size * self._lava_location[1] + 2))

            if self._robot_display:
                canvas.blit(robot,
                            (pix_square_size * self._agent_location[0], pix_square_size * self._agent_location[1]))

            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()