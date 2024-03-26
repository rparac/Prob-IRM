import copy
import datetime as dt
import json
import os
import warnings
from collections import defaultdict
from typing import Dict, List

import joblib
# import joblib
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw

from rm_marl.utils.logging import create_rm_state_logs, create_learnt_rm_logs


class Trainer:
    def __init__(self, local_envs: dict, shared_envs: dict, agents: dict):
        self.envs = local_envs or shared_envs
        self.testing_envs = shared_envs
        self.agents = agents

        self.total_steps = 0
        self.test_episode = 0

        # Logs for plots more that SummaryWriter can't represent, so we use matplotlib + SummaryWritter add image
        # Keeps track of all rm states in each environment. Used for logging of state transition diagram.
        self.all_recorded_rm_states: Dict[str, set] = defaultdict(set)
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
            result = self._run(self.envs, run_config, logger)
        except KeyboardInterrupt:
            result = None

        if run_config["extra_debug_information"]:
            create_rm_state_logs(log_dir, logger, run_config["total_episodes"], self.test_episode,
                                 run_config["testing_freq"], self.last_timestep_train_info,
                                 self.last_timestep_test_info,
                                 self.all_recorded_rm_states, self.rm_relearned_episodes)
            create_learnt_rm_logs(log_dir, logger)

        _ = [e.close() for e in self.envs.values()]
        logger.close()
        if run_config["training"]:
            self.save(log_dir)

        return result

    def _run(self, envs: dict, run_config: dict, logger: SummaryWriter):
        base_seed = run_config["seed"]

        steps = defaultdict(list)
        losses = defaultdict(list)
        rewards = defaultdict(list)
        shaping_rewards = defaultdict(list)
        successes = defaultdict(int)
        failures = defaultdict(int)
        timeouts = defaultdict(int)

        _ = [a.set_log_folder(os.path.join(logger.log_dir, aid)) for aid, a in self.agents.items()]

        for episode in tqdm(range(1, 1 + run_config["total_episodes"])):
            if not run_config["training"]:
                self.test_episode += 1

            episode_losses = defaultdict(list)
            episode_frames = defaultdict(list)
            episode_shaping_rewards = defaultdict(list)

            # seperate for each env_id
            last_timestep_in_u = {}

            seed = base_seed + episode - 1

            # reset and initial setup
            dones = {env_id: False for env_id in envs.keys()}
            env_agents = {}

            _ = [a.reset(agent_id=aid) for aid, a in self.agents.items()]

            obs, infos, env_agents, shared_events = {}, {}, {}, {}
            for env_id, env in envs.items():
                o, i = env.reset()
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

                    if dones[env_id]:
                        continue

                    # env.render with pygame does not work in a headless mode
                    if episode % run_config["recording_freq"] == 0 and not run_config["no_display"]:
                        episode_frames[env_id].append(env.render())

                    actions = {
                        aid: a.action(
                            self._project_obs(obs[env_id], a, aid),
                            greedy=run_config.get("greedy", True)
                            if not run_config["training"]
                            else False,
                            testing=not run_config["training"],
                            agent_id=aid,
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
                    if "rm_state" in info:
                        most_likely_state = info["rm_state"]
                        if isinstance(most_likely_state, np.ndarray):
                            # TODO: not working with multi agent case
                            most_likely_state_idx = np.argmax(most_likely_state)
                            curr_a = list(env_agents[env_id].values())[0]
                            most_likely_state = curr_a.rm.states[most_likely_state_idx]
                        last_timestep_in_u[env_id][most_likely_state] = steps_count

                    if "shaping_reward" in info:
                        episode_shaping_rewards[env_id].append(info["shaping_reward"])

                    # update the agent's RM and Q-functions
                    agent_loss = []
                    interrupt_episode = terminated or truncated
                    for aid, a in env_agents[env_id].items():
                        current_u = a.get_current_state(agent_id=aid)
                        loss, agents_to_interrupt, updated_rm = a.update_agent(
                            self._project_obs(obs[env_id], a, aid),
                            actions[aid],
                            reward,
                            terminated,
                            truncated,
                            info.get("is_positive_trace", True),
                            self._project_obs(next_obs, a, aid),
                            synchronized_labels[aid],
                            learning=run_config["training"],
                            agent_id=aid,
                        )

                        if updated_rm is not None:
                            curr_relearned_episodes = self.rm_relearned_episodes.get(env_id, [])
                            curr_relearned_episodes.append(episode)
                            self.rm_relearned_episodes[env_id] = curr_relearned_episodes

                            if "shaping_reward" in info:
                                env.set_shaping_rm(updated_rm)

                        if run_config["training"]:
                            if agents_to_interrupt:
                                interrupt_episode = True
                                info["episode"] = {
                                    "l": env.episode_lengths[0],
                                    "r": env.episode_returns[0]
                                }
                                for _env_id, ag_dict in env_agents.items():
                                    # There is an agent that should be interrupted
                                    if agents_to_interrupt.intersection(set(ag_dict.keys())) != set():
                                        dones[_env_id] = True
                                        infos[_env_id]["episode"] = {
                                            "l": envs[_env_id].episode_lengths[0],
                                            "r": envs[_env_id].episode_returns[0]
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
                                    aid,
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

            # print('yes')
            # track metrics and log them in TB
            for env_id in envs.keys():
                prefix = "training" if run_config["training"] else "eval"

                episode_reward = infos[env_id]["episode"]["r"]
                episode_length = infos[env_id]["episode"]["l"]

                steps[env_id].append(episode_length)
                rewards[env_id].append(episode_reward)

                if episode_reward == 1:
                    successes[env_id] += 1
                elif episode_reward == -1:
                    failures[env_id] += 1
                else:
                    timeouts[env_id] += 1

                assert successes[env_id] + failures[env_id] + timeouts[env_id] == episode, "Something is wrong"

                if episode_losses[env_id]:
                    losses[env_id].append(np.mean(episode_losses[env_id]))

                if episode_shaping_rewards[env_id]:
                    shaping_rewards[env_id] = episode_shaping_rewards[env_id]

                if episode % run_config["log_freq"] == 0:

                    # Loss information
                    if losses[env_id]:
                        logger.add_scalar(
                            f"{prefix}/loss/{env_id}", np.mean(losses[env_id]), self.total_steps
                        )

                    # Reward shaping information
                    if shaping_rewards[env_id]:
                        np_data = np.array(shaping_rewards[env_id])
                        unique, counts = np.unique(np_data, return_counts=True)
                        freq_strings = [f"{value}: #{freq} | " for value, freq in zip(unique, counts)]
                        logged_text = "  \n".join(freq_strings)
                        logger.add_text(
                            f"{prefix}/reward_shaping/frequencies/{env_id}", str(logged_text),
                            episode if run_config["training"] else self.test_episode
                        )

                        logged_text = "  \n".join([f"{i}: {value}" for i, value in enumerate(shaping_rewards[env_id])])
                        logger.add_text(
                            f"{prefix}/reward_shaping/history/{env_id}", logged_text,
                            episode if run_config["training"] else self.test_episode
                        )
                        logger.add_histogram(
                            f"{prefix}/reward_shaping/{env_id}", np_data,
                            episode if run_config["training"] else self.test_episode
                        )

                    # Episode number of steps
                    logger.add_scalar(
                        f"{prefix}/num_steps/{env_id}", steps[env_id][-1],
                        episode if run_config["training"] else self.test_episode
                    )

                    # Episode reward
                    logger.add_scalar(
                        f"{prefix}/reward/{env_id}", rewards[env_id][-1],
                        episode if run_config["training"] else self.test_episode
                    )

                    # Success/Failure/Timeouts rate
                    logger.add_scalar(
                        f"{prefix}/success_rate/{env_id}", successes[env_id] / episode,
                        episode if run_config["training"] else self.test_episode
                    )
                    logger.add_scalar(
                        f"{prefix}/failure_rate/{env_id}", failures[env_id] / episode,
                        episode if run_config["training"] else self.test_episode
                    )
                    logger.add_scalar(
                        f"{prefix}/timeout_rate/{env_id}", timeouts[env_id] / episode,
                        episode if run_config["training"] else self.test_episode
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
                    video = self._add_steps_to_replay(video)
                    logger.add_video(
                        f"{prefix}/replay/{env_id}", video,
                        episode if run_config["training"] else self.test_episode
                    )

            if run_config["training"] and episode % run_config["testing_freq"] == 0:
                self._run(self.testing_envs, {
                    "training": False,
                    "log_freq": 1,
                    "no_display": run_config["no_display"],
                    "recording_freq": 1,
                    "total_episodes": 1,
                    "greedy": run_config.get("greedy", True),
                    "seed": run_config["seed"],
                    "synchronize": run_config["synchronize"],
                }, logger)

        # TODO: make cleaner
        # Sums the rewards of the last 100 episodes
        return sum(rewards[list(self.testing_envs.keys())[0]][run_config["total_episodes"] - 100:])

    @staticmethod
    def _add_steps_to_replay(video_data):
        """
        Expand a replay to visualize the step number of each frame.

        Aside from being some nice quality-of-life, this method is also needed due to a quirk in the
        inner functioning of SummaryWriter.add_video(). Since Tensorboard, at the time of writing, does not
        *really* support video data, a workaround is used instead, consisting in adding video data by means
        of animated GIF images. For some reason, when such a GIF is created, if a number of consecutive frames
        is identical to each other, the resulting GIF will contain only one copy of the frame.

        Parameters
        ----------
        video_data Numpy array with shape (1, num_frames, n_channels, video_height, video_width)

        Returns
        -------
        Numpy array containing the replay and steps information

        """

        num_frames, channels, video_h, video_w = video_data.shape[1:]
        steps_bar_h = int(video_h / 5)

        steps_bar = np.zeros((1, num_frames, channels, steps_bar_h, video_w), dtype=video_data.dtype)
        full_video = np.concatenate((steps_bar, video_data), axis=3)

        for i in range(num_frames):
            transposed_steps_bar = np.transpose(steps_bar[0, i], axes=(1, 2, 0))
            with Image.fromarray(transposed_steps_bar, mode='RGB') as steps_bar_img:
                drawer = ImageDraw.Draw(steps_bar_img)
                drawer.text((0, 0), f"# Steps: {i}", font_size=54, fill='#ffffff')
                steps_bar_data = np.transpose(np.asarray(steps_bar_img), axes=(2, 0, 1))
                full_video[0, i, :, 0:steps_bar_h, :] = steps_bar_data

        return full_video

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
    def _counterfactual_update(env, agent, state, current_u, action, done, next_state, aid):
        labels = env.get_labels(next_state, state)

        if isinstance(labels, dict):
            labels = [lbl for lbl, prob in labels.items() if prob > 0.5]
            raise NotImplementedError("Need to implement Counterfactual reasoning")

        for u in agent.rm.states:
            if u != current_u and not agent.rm.is_state_terminal(u):
                l = env.filter_labels(labels, u)

                next_u = agent.rm.get_next_state(u, l)
                r = agent.rm.get_reward(u, next_u)

                agent.learn(state, u, action, r, done, next_state, next_u, agent_id=aid)

    def save(self, path):
        trainer_path = os.path.join(path, "trainer.pkl")
        joblib.dump(self, trainer_path)
        print(f"Trainer saved at: {trainer_path}")

    @classmethod
    def load(cls, path):
        if "trainer.pkl" not in path:
            path = os.path.join(path, "trainer.pkl")
        return joblib.load(path)
