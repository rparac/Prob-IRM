import copy
import datetime as dt
import os
from collections import defaultdict

import json
from typing import Dict, Any, List

import joblib

# import joblib
import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings


class Trainer:
    def __init__(self, local_envs: dict, shared_envs: dict, agents: dict):
        self.envs = local_envs or shared_envs
        self.testing_envs = shared_envs
        self.agents = agents

        self.total_steps = 0
        self.test_episodes = 0

        # Logs for plots more that SummaryWriter can't represent, so we use matplotlib + SummaryWritter add image

        # Keeps track of all rm states in each environment. Used for logging of state transition diagram.
        self.all_recorded_rm_states: Dict[str, set] = dict()
        # Stores for each episode in an environment a dictionary of (RM state, timestep) pairs
        # from env_id -> [{u -> last_timestep}]
        self.last_timestep_train_info: Dict[str, List[Dict[str, int]]] = {}
        self.last_timestep_test_info: Dict[str, List[Dict[str, int]]] = {}

        # Contains episodes when a RM was relearned:
        #  Dict[env_id -> List[episode_when_relearned]]
        self.rm_relearned_episodes: Dict[str, List[int]] = {}

    def run(self, run_config: dict):
        log_dir = os.path.join(
            run_config["log_dir"],
            run_config["name"],
            dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        logger = SummaryWriter(log_dir)

        config_path = os.path.join(log_dir, "run_config.json")
        with open(config_path, 'w') as f:
            json.dump(dict(run_config), f, indent=4)

        try:
            self._run(self.envs, run_config, logger)
        except KeyboardInterrupt:
            pass

        self.create_rm_state_logs(logger, log_dir, run_config["total_episodes"], run_config["testing_freq"])

        _ = [e.close() for e in self.envs.values()]
        logger.close()
        if run_config["training"]:
            self.save(log_dir)

    def _run(self, envs: dict, run_config: dict, logger: SummaryWriter):
        base_seed = run_config["seed"]

        steps = defaultdict(list)
        losses = defaultdict(list)
        rewards = defaultdict(list)

        self.all_recorded_rm_states = self.all_recorded_rm_states or {env_id: set() for env_id in envs.keys()}

        _ = [a.set_log_folder(os.path.join(logger.log_dir, aid)) for aid, a in self.agents.items()]

        for episode in tqdm(range(1, 1 + run_config["total_episodes"])):
            if not run_config["training"]:
                self.test_episodes += 1

            episode_losses = defaultdict(list)
            episode_frames = defaultdict(list)

            # seperate for each env_id
            last_timestep_in_u = {}

            seed = base_seed + episode - 1

            # reset and initial setup
            dones = {env_id: False for env_id in envs.keys()}
            env_agents = {}

            _ = [a.reset(seed=seed) for a in self.agents.values()]

            obs, infos, env_agents, shared_events = {}, {}, {}, {}
            for env_id, env in envs.items():
                o, i = env.reset(seed=seed)
                obs[env_id] = o
                infos[env_id] = i

                env_agents[env_id] = {
                    aid: a for aid, a in self.agents.items() if aid in obs[env_id]
                }
                shared_events[env_id] = self._get_shared_events(env_agents[env_id])
                last_timestep_in_u[env_id] = {}

            steps_count = 0
            while not all(dones.values()):
                steps_count += 1

                if run_config["training"]:
                    self.total_steps += 1

                for env_id, env in envs.items():

                    if episode % run_config["recording_freq"] == 0:
                        episode_frames[env_id].append(env.render())

                    if dones[env_id]:
                        continue

                    actions = {
                        aid: a.action(
                            self._project_obs(obs[env_id], a, aid),
                            greedy=run_config.get("greedy", True)
                            if not run_config["training"]
                            else False,
                        )
                        for aid, a in env_agents[env_id].items()
                    }
                    next_obs, reward, terminated, truncated, info = env.step(actions)

                    done = terminated or truncated

                    labels = info["labels"]
                    agent_labels = {}
                    _ = [agent_labels.update(self._project_labels(labels, a, aid)) for aid, a in
                         env_agents[env_id].items()]

                    if run_config["synchronize"]:
                        synchronized_labels = self._synchronize(shared_events[env_id], agent_labels)
                        assert all(agent_labels[aid] == synchronized_labels[aid] for aid in env_agents[
                            env_id].keys()), f"Not synchronized!! {agent_labels}, {synchronized_labels}"
                    else:
                        synchronized_labels = agent_labels

                    # track state metric for logging
                    last_timestep_in_u[env_id][info["rm_state"]] = steps_count

                    # update the agent's RM and Q-functions
                    agent_loss = []
                    interrupt_episode = terminated or truncated
                    for aid, a in env_agents[env_id].items():
                        current_u = a.u
                        loss, interrupt, rm_updated = a.update_agent(
                            self._project_obs(obs[env_id], a, aid),
                            actions[aid],
                            reward,
                            terminated,
                            truncated,
                            self._project_obs(next_obs, a, aid),
                            synchronized_labels[aid],
                            learning=run_config["training"],
                        )

                        if rm_updated:
                            curr_relearned_episodes = self.rm_relearned_episodes.get(env_id, [])
                            curr_relearned_episodes.append(episode)
                            self.rm_relearned_episodes[env_id] = curr_relearned_episodes

                        if run_config["training"]:
                            if interrupt:
                                interrupt_episode = True
                                info["episode"] = {
                                    "l": env.episode_lengths[0],
                                    "r": env.episode_returns[0]
                                }
                                break

                            agent_loss.append(loss)
                            if run_config["counterfactual_update"] and not interrupt_episode:
                                self._counterfactual_update(
                                    env,
                                    a,
                                    self._project_obs(obs[env_id], a, aid),
                                    current_u,
                                    actions[aid],
                                    done,
                                    self._project_obs(next_obs, a, aid),
                                )

                    obs[env_id] = next_obs
                    infos[env_id] = info
                    dones[env_id] = interrupt_episode

                    if agent_loss:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            mean_loss = np.nanmean(agent_loss)
                        if not np.isnan(mean_loss):
                            episode_losses[env_id].append(mean_loss)

            # track metrics and log them in TB
            for env_id in envs.keys():
                prefix = "training" if run_config["training"] else "eval"

                steps[env_id].append(infos[env_id]["episode"]["l"])
                rewards[env_id].append(infos[env_id]["episode"]["r"])
                if episode_losses[env_id]:
                    losses[env_id].append(np.mean(episode_losses[env_id]))

                if episode % run_config["log_freq"] == 0:
                    if losses[env_id]:
                        logger.add_scalar(
                            f"{prefix}/loss/{env_id}", np.mean(losses[env_id]), self.total_steps
                        )
                    logger.add_scalar(
                        f"{prefix}/num_steps/{env_id}", np.mean(steps[env_id]), self.total_steps
                    )
                    logger.add_scalar(
                        f"{prefix}/reward/{env_id}", np.mean(rewards[env_id]), self.total_steps
                    )
                    self.all_recorded_rm_states[env_id] = self.all_recorded_rm_states[env_id].union(
                        last_timestep_in_u[env_id].keys())

                    if run_config["training"]:
                        timestep_train_info = self.last_timestep_train_info.get(env_id, [])
                        timestep_train_info.append(last_timestep_in_u[env_id])
                        self.last_timestep_train_info[env_id] = timestep_train_info
                    else:
                        timestep_test_info = self.last_timestep_test_info.get(env_id, [])
                        timestep_test_info.append(last_timestep_in_u[env_id])
                        self.last_timestep_test_info[env_id] = timestep_test_info

                if episode_frames[env_id]:
                    video = np.array(episode_frames[env_id]).transpose(0, 3, 1, 2)[
                            np.newaxis, :
                            ]
                    logger.add_video(f"{prefix}/replay/{env_id}", video, self.total_steps)

            if run_config["training"] and episode % run_config["testing_freq"] == 0:
                self._run(self.testing_envs, {
                    "training": False,
                    "log_freq": 1,
                    "recording_freq": 1,
                    "total_episodes": 1,
                    "greedy": run_config.get("greedy", True),
                    "seed": run_config["seed"],
                    "synchronize": run_config["synchronize"],
                }, logger)

    @staticmethod
    def _project_labels(labels, a, aid):
        return {
            aid: a.project_labels(labels)
        }

    @staticmethod
    def _project_obs(obs, _a, aid):
        return {
            i: o for i, o in obs.items() if i == aid
        }

    @staticmethod
    def _get_shared_events(env_agents):
        shared_events = {}
        for aid1, agent1 in env_agents.items():
            for aid2, agent2 in env_agents.items():
                if aid1 >= aid2:
                    continue
                intersection = set(agent1.rm.get_valid_events()).intersection(agent2.rm.get_valid_events())
                shared_events[tuple(sorted([aid1, aid2]))] = intersection
        return shared_events

    @staticmethod
    def _synchronize(shared_events, agent_labels):
        agent_labels = copy.deepcopy(agent_labels)

        for (aid1, aid2), events in shared_events.items():
            for e in events:
                if not all(e in agent_labels[aid] for aid in (aid1, aid2)):
                    agent_labels[aid1] = tuple(l for l in agent_labels[aid1] if l != e)
                    agent_labels[aid2] = tuple(l for l in agent_labels[aid2] if l != e)

        return agent_labels

    @staticmethod
    def _counterfactual_update(env, agent, state, current_u, action, done, next_state):
        labels = env.get_labels(next_state, state)

        for u in agent.rm.states:
            if u != current_u and not agent.rm.is_state_terminal(u):
                l = env.filter_labels(labels, u)

                next_u = agent.rm.get_next_state(u, l)
                r = agent.rm.get_reward(u, next_u)

                agent.learn(state, u, action, r, done, next_state, next_u)

    # SummaryWriter alone was not general enough to create an image with multiple lines and vertical
    # lines signifying when we relearned the RM.
    def create_rm_state_logs(self, logger: SummaryWriter, log_dir: str, total_episodes: int, testing_freq: int):
        # train image generation
        self._create_rm_state_logs(total_episodes, log_dir, self.last_timestep_train_info, is_test_log=False)
        # test image generation
        self._create_rm_state_logs(self.test_episodes, log_dir, self.last_timestep_test_info, is_test_log=True,
                                   testing_freq=testing_freq)

    def _create_rm_state_logs(self, n_episodes: int, log_dir: str, last_timestep_info: dict, is_test_log: bool,
                              testing_freq: int=1):
        for env_id, test_dicts in last_timestep_info.items():
            x_values = np.arange(1, n_episodes + 1)
            fig_width, fig_height = max(len(x_values) / 8, 6), 6
            plt.figure(f"{env_id}{'_test' if is_test_log else ''}", figsize=(fig_width, fig_height))
            for u in self.all_recorded_rm_states[env_id]:
                y_values = [u_timestep_dict.get(u, None) for u_timestep_dict in test_dicts]
                y_values = np.array(y_values)

                plt.plot(x_values, y_values, label=f"u{u}", marker='o', markersize=3)

            for relearn_episode in self.rm_relearned_episodes.get(env_id, []):
                first_ep_with_new_rm = ((relearn_episode - 1) // testing_freq) + 1
                # Moving the line for better readability
                first_ep_with_new_rm -= 0.1
                plt.axvline(x=first_ep_with_new_rm, color='r', linestyle='--')

            plt.xlabel('episode')
            plt.ylabel('last timestep in state u')
            plt.legend()
            img_path = f"{log_dir}/state_transition_{'test' if is_test_log else 'train'}_{env_id}"
            plt.savefig(img_path)
            # TODO: the line below fails https://github.com/pytorch/pytorch/issues/24175
            #  we should try out to see if it does work on another machine
            # logger.add_image(f"training/last_timestep_in_rm_state/{env_id}", img_path)

    def save(self, path):
        trainer_path = os.path.join(path, "trainer.pkl")
        joblib.dump(self, trainer_path)
        print(f"Trainer saved at: {trainer_path}")

    @classmethod
    def load(cls, path):
        if "trainer.pkl" not in path:
            path = os.path.join(path, "trainer.pkl")
        return joblib.load(path)
