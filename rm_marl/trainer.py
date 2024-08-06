import copy
import datetime as dt
import json
import os
import shutil
import warnings
from collections import defaultdict
from typing import Dict, List, Union, DefaultDict, Any

import joblib
# import joblib
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw

from rm_marl.utils.logging import create_rm_state_logs, create_learnt_rm_logs
from rm_marl.utils.trainer_utils import TrainState


class Trainer:
    def __init__(self, envs: dict, agents: dict, env_config: DictConfig = None):
        self.envs = envs
        self.testing_envs = envs
        self.agents = agents

        # Stores configuration used to run this experiment
        if env_config is not None:
            self.env_config = OmegaConf.to_container(env_config)
        else:
            self.env_config = None

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

        # The results are aggregated based on the last num_episodes_for_aggregation
        self.num_episodes_for_aggregation = 100

    def get_log_dir(self, run_config: Union[dict, DictConfig]) -> str:
        base_dir = os.path.join(
            run_config["log_dir"],
            run_config["name"],
        )
        checkpointed_train_state = None
        if run_config["restart_from_checkpoint"] and os.path.isdir(base_dir):
            # Get the last run with the same name
            log_dir = os.path.join(
                base_dir,
                max(os.listdir(base_dir)),
            )
        else:
            log_dir = os.path.join(
                base_dir,
                dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
        return str(log_dir)

    def run(self, run_config: Union[dict, DictConfig], trial=None):
        log_dir = self.get_log_dir(run_config)
        logger = SummaryWriter(log_dir)

        checkpointed_train_state = self.load_checkpoint(log_dir)

        config_path = os.path.join(log_dir, "run_config.json")
        with open(config_path, 'w') as f:
            # DictConfig can't be serialized
            if isinstance(run_config, DictConfig):
                run_config = OmegaConf.to_container(run_config)
            json.dump(dict(run_config), f, indent=4)

        if self.env_config is not None:
            with open(os.path.join(log_dir, "env_config.json"), 'w') as f:
                json.dump(dict(self.env_config), f, indent=4)

        try:
            result = self._run(self.envs, run_config, logger, checkpointed_train_state, trial)
        except KeyboardInterrupt:
            result = None

        if run_config["extra_debug_information"]:
            create_learnt_rm_logs(log_dir, logger)

        _ = [e.close() for e in self.envs.values()]
        logger.close()
        if run_config["training"]:
            self.save(log_dir)

        return result

    def _run(self, envs: dict, run_config: dict, logger: SummaryWriter,
             train_state=None, trial=None):

        train_state = train_state or TrainState()

        _ = [a.set_log_folder(os.path.join(logger.log_dir, aid)) for aid, a in self.agents.items()]

        for episode in tqdm(range(train_state.episodes_completed + 1, 1 + run_config["total_episodes"]),
                            initial=train_state.episodes_completed + 1,
                            total=run_config["total_episodes"]):

            self._track_algo_metrics(episode, logger, run_config)

            if not run_config["training"]:
                self.test_episode += 1

            episode_losses = defaultdict(list)
            episode_frames = defaultdict(list)
            episode_shaping_rewards = defaultdict(list)

            # seperate for each env_id
            last_timestep_in_u = {}

            # reset and initial setup
            dones = {env_id: False for env_id in envs.keys()}

            _ = [a.reset(agent_id=aid) for aid, a in self.agents.items()]

            obs, infos, env_agents = {}, {}, {}
            for env_id, env in envs.items():
                o, i = env.reset()
                obs[env_id] = o
                infos[env_id] = i

                env_agents[env_id] = {
                    aid: a for aid, a in self.agents.items() if aid in obs[env_id]
                }
                last_timestep_in_u[env_id] = {}

            steps_count = 0
            while not all(dones.values()):
                steps_count += 1

                if run_config["training"]:
                    self.total_steps += 1

                for env_id, env in envs.items():

                    if dones[env_id]:
                        continue

                    # Render a frame if there is a display
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
                            labels=infos[env_id]['labels'],
                        )
                        for aid, a in env_agents[env_id].items()
                    }
                    next_obs, reward, terminated, truncated, info = env.step(actions)

                    labels = info["labels"]
                    agent_labels = {}
                    for aid, a in env_agents[env_id].items():
                        agent_labels.update(self._project_labels(labels, a, aid))

                    # track state metric for logging
                    if "rm_state" in info:
                        most_likely_state = info["rm_state"]
                        if isinstance(most_likely_state, np.ndarray):
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
                        loss, agents_to_interrupt, updated_rm = a.update_agent(
                            self._project_obs(obs[env_id], a, aid),
                            actions[aid],
                            reward,
                            terminated,
                            truncated,
                            info.get("is_positive_trace", True),
                            self._project_obs(next_obs, a, aid),
                            labels=agent_labels[aid],
                            learning=run_config["training"],
                            agent_id=aid,
                        )

                        if updated_rm is not None:
                            curr_relearned_episodes = self.rm_relearned_episodes.get(env_id, [])
                            curr_relearned_episodes.append(episode)
                            self.rm_relearned_episodes[env_id] = curr_relearned_episodes

                            if "shaping_reward" in info:
                                for _env_id in self.env_ids_to_interrupt(env_agents, agents_to_interrupt):
                                    envs[_env_id].set_shaping_rm(updated_rm)

                        if run_config["training"]:
                            if agents_to_interrupt:

                                if aid in agents_to_interrupt:
                                    interrupt_episode = True
                                    info["episode"] = {
                                        "l": env.episode_lengths[0],
                                        "r": env.episode_returns[0]
                                    }
                                    agents_to_interrupt.remove(aid)

                                for _env_id in self.env_ids_to_interrupt(env_agents, agents_to_interrupt):
                                    dones[_env_id] = True
                                    infos[_env_id]["episode"] = {
                                        "l": envs[_env_id].episode_lengths[0],
                                        "r": envs[_env_id].episode_returns[0]
                                    }

                                break

                            agent_loss.append(loss)

                    obs[env_id] = next_obs
                    infos[env_id] = info
                    dones[env_id] = interrupt_episode

                    if agent_loss:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            mean_loss = np.nanmean(agent_loss)
                        if not np.isnan(mean_loss):
                            episode_losses[env_id].append(mean_loss)

            self._track_metrics(envs, train_state, episode, episode_frames, episode_losses, episode_shaping_rewards,
                                logger, infos, last_timestep_in_u, run_config)

            # Report metric to optuna for early stopping
            if trial and run_config["training"] and episode > self.num_episodes_for_aggregation:
                trial.report(self._compute_score(train_state.steps), episode)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Store a checkpoint
            if episode % run_config["checkpoint_freq"] == 0:
                new_state = copy.deepcopy(train_state)
                new_state.episode = episode
                self.save_checkpoint(logger.log_dir, train_state)

            # Run test
            if run_config["training"] and episode % run_config["testing_freq"] == 0:
                self._run(self.testing_envs, {
                    "training": False,
                    "log_freq": 1,
                    "no_display": run_config["no_display"],
                    "recording_freq": 1,
                    "total_episodes": 1,
                    "greedy": run_config.get("greedy", True),
                    "seed": run_config["seed"],
                    "checkpoint_freq": run_config["checkpoint_freq"],
                    "only_log_base_metrics": run_config["only_log_base_metrics"]
                }, logger)

        return self._compute_score(train_state.steps)

    def _track_algo_metrics(self, episode, logger, run_config):
        if (run_config["training"] and run_config["extra_debug_information"] and
                not run_config["only_log_base_metrics"]):
            for aid, a in self.agents.items():
                if hasattr(a, 'rm_agents'):
                    algo_stats = a.rm_agents[aid].algo.get_statistics()
                else:
                    algo_stats = a.algo.get_statistics()
                # Using rm_learner stats
                agent_stats = a.get_statistics()
                logger.add_scalar(f'algo/{aid}/policy_age', algo_stats["policy_age"], episode)
                logger.add_scalar(f'algo/{aid}/epsilon', algo_stats["epsilon"], episode)
                for stat_key, stat_value in agent_stats.items():
                    logger.add_scalar(f'agent/{stat_key}', stat_value, episode)

    def _track_metrics(self, envs, train_state, episode, episode_frames, episode_losses, episode_shaping_rewards,
                       logger, infos, last_timestep_in_u, run_config):
        # track metrics and log them in TB
        for env_id in envs.keys():
            prefix = "training" if run_config["training"] else "eval"

            episode_reward = infos[env_id]["episode"]["r"]
            episode_length = infos[env_id]["episode"]["l"]

            train_state.steps[env_id].append(episode_length)
            train_state.rewards[env_id].append(episode_reward)

            if len(train_state.cumulative_steps[env_id]) > 0:
                total_steps_so_far = episode_length + train_state.cumulative_steps[env_id][-1]
            else:
                total_steps_so_far = episode_length
            train_state.cumulative_steps[env_id].append(total_steps_so_far)

            if episode_reward == 1:
                train_state.successes[env_id] += 1
            elif episode_reward == -1:
                train_state.failures[env_id] += 1
            else:
                train_state.timeouts[env_id] += 1

            assert train_state.successes[env_id] + train_state.failures[env_id] + train_state.timeouts[
                env_id] == episode, "Something is wrong"

            if episode_losses[env_id]:
                train_state.losses[env_id].append(np.mean(episode_losses[env_id]))

            if episode_shaping_rewards[env_id]:
                train_state.shaping_rewards[env_id] = episode_shaping_rewards[env_id]

            if episode % run_config["log_freq"] == 0:

                # Episode reward
                logger.add_scalar(
                    f"{prefix}/reward/{env_id}", train_state.rewards[env_id][-1],
                    episode if run_config["training"] else self.test_episode
                )

                # Number of steps taken by the agent in each episode
                logger.add_scalar(
                    f"{prefix}/num_steps/{env_id}", train_state.steps[env_id][-1],
                    episode if run_config["training"] else self.test_episode
                )

                # Skip over the rest of the logging if the user requested only basic metrics
                if run_config["only_log_base_metrics"]:
                    continue

                # Loss information
                if train_state.losses[env_id]:
                    logger.add_scalar(
                        f"{prefix}/loss/{env_id}", np.mean(train_state.losses[env_id]), self.total_steps
                    )

                # Reward shaping information
                if train_state.shaping_rewards[env_id]:
                    np_data = np.array(train_state.shaping_rewards[env_id])
                    unique, counts = np.unique(np_data, return_counts=True)
                    freq_strings = [f"{value}: #{freq} | " for value, freq in zip(unique, counts)]
                    logged_text = "  \n".join(freq_strings)
                    logger.add_text(
                        f"{prefix}/reward_shaping/frequencies/{env_id}", str(logged_text),
                        episode if run_config["training"] else self.test_episode
                    )

                    logged_text = "  \n".join(
                        [f"{i}: {value}" for i, value in enumerate(train_state.shaping_rewards[env_id])])
                    logger.add_text(
                        f"{prefix}/reward_shaping/history/{env_id}", logged_text,
                        episode if run_config["training"] else self.test_episode
                    )
                    logger.add_histogram(
                        f"{prefix}/reward_shaping/{env_id}", np_data,
                        episode if run_config["training"] else self.test_episode
                    )

                # Cumulative number of steps so far, among all episodes
                # Only logged during training, as in evaluation we only care about the steps taken in each episode
                if run_config["training"]:
                    logger.add_scalar(
                        f"{prefix}/tot_steps/{env_id}", train_state.cumulative_steps[env_id][-1],
                        episode
                    )

                # Success/Failure/Timeouts rate
                # At test time, they are either 0/1 while at training they are the rate of
                # success up until a specific episode number
                logger.add_scalar(
                    f"{prefix}/success_rate/{env_id}", train_state.successes[env_id] / episode,
                    episode if run_config["training"] else self.test_episode
                )
                logger.add_scalar(
                    f"{prefix}/failure_rate/{env_id}", train_state.failures[env_id] / episode,
                    episode if run_config["training"] else self.test_episode
                )
                logger.add_scalar(
                    f"{prefix}/timeout_rate/{env_id}", train_state.timeouts[env_id] / episode,
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
                video = self._improve_replay(video, success=episode_reward == 1)
                logger.add_video(
                    f"{prefix}/replay/{env_id}", video,
                    episode if run_config["training"] else self.test_episode
                )

    # Computes the sum of a given metric for the last num_episodes
    def _compute_score(self, metric):
        total = 0
        for _env in self.testing_envs.keys():
            # Number of steps
            total += sum(metric[_env][-self.num_episodes_for_aggregation:])

        return total / len(self.testing_envs)

    @staticmethod
    def _improve_replay(video_data, *, success):
        """
        Expand a replay to visualize additional information..

        Aside from being some nice quality-of-life, this method is also needed due to a quirk in the
        inner functioning of SummaryWriter.add_video(). Since Tensorboard, at the time of writing, does not
        *really* support video data, a workaround is used instead, consisting in adding video data by means
        of animated GIF images. For some reason, when such a GIF is created, if a number of consecutive frames
        is identical to each other, the resulting GIF will contain only one copy of the frame.

        Parameters
        ----------
        video_data Numpy array with shape (1, num_frames, n_channels, video_height, video_width)
        success True if the agent was able to succesfully complete the task

        Returns
        -------
        Numpy array containing the replay with additional information

        """

        num_frames, channels, video_h, video_w = video_data.shape[1:]
        info_bar_h = int(video_h / 5)

        info_bar = np.zeros((1, num_frames, channels, info_bar_h, video_w), dtype=video_data.dtype)
        full_video = np.concatenate((info_bar, video_data), axis=3)

        for i in range(num_frames):
            transposed_info_bar = np.transpose(info_bar[0, i], axes=(1, 2, 0))
            with Image.fromarray(transposed_info_bar, mode='RGB') as info_bar_img:
                drawer = ImageDraw.Draw(info_bar_img)
                drawer.text((0, 0), f"# Steps: {i + 1}/{num_frames}", font_size=50, fill='#ffffff')
                drawer.text((0, 50), f"Success: {'Yes' if success else 'No'}", font_size=32, fill='#ffffff')
                info_bar_data = np.transpose(np.asarray(info_bar_img), axes=(2, 0, 1))
                full_video[0, i, :, 0:info_bar_h, :] = info_bar_data

        return full_video

    @staticmethod
    def env_ids_to_interrupt(env_agents, agents_to_interrupt):
        for _env_id, ag_dict in env_agents.items():
            # There is an agent that should be interrupted
            if agents_to_interrupt.intersection(set(ag_dict.keys())) != set():
                yield _env_id

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

    def save_checkpoint(self, path, train_state: TrainState):
        checkpoint_data = {
            "trainer": self,
            "train_state": train_state,
        }

        checkpoint_path_temp = os.path.join(path, "temp_checkpoint.pkl")
        joblib.dump(checkpoint_data, checkpoint_path_temp)
        checkpoint_path = os.path.join(path, "checkpoint.pkl")
        # Move is used to reduce the chance of being in an inconsistent state (due to an interrupt)
        shutil.move(checkpoint_path_temp, checkpoint_path)

    def load_checkpoint(self, path):
        if "checkpoint.pkl" not in path:
            path = os.path.join(path, "checkpoint.pkl")
        if not os.path.exists(path):
            return None

        last_run = joblib.load(path)
        # Replace the state dictionary
        self.__dict__ = last_run["trainer"].__dict__
        return last_run["train_state"]

    def save(self, path):
        trainer_path = os.path.join(path, "trainer.pkl")
        joblib.dump(self, trainer_path)
        print(f"Trainer saved at: {trainer_path}")

    @classmethod
    def load(cls, path):
        if "trainer.pkl" not in path:
            path = os.path.join(path, "trainer.pkl")
        return joblib.load(path)
