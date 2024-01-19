import logging
import os
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt


def getLogger(name):
    logger = logging.getLogger(name.rsplit('.', 1)[-1])
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    return logger


# Creates a diagram tracking the agent's last step in each RM step.
# Useful when debugging whether the agent indeed reaches expected RM states.
def create_rm_state_logs(log_dir: str, total_episodes: int, test_episodes: int, testing_freq: int,
                         last_timestep_train_info: Dict[str, List[Dict[str, int]]],
                         last_timestep_test_info: Dict[str, List[Dict[str, int]]],
                         all_recorded_rm_states: Dict[str, set],
                         rm_relearned_episodes: Dict[str, List[int]]):
    # train image generation
    _create_rm_state_logs(total_episodes, log_dir, last_timestep_train_info, all_recorded_rm_states,
                          rm_relearned_episodes,
                          is_test_log=False)
    # test image generation
    _create_rm_state_logs(test_episodes, log_dir, last_timestep_test_info, all_recorded_rm_states,
                          rm_relearned_episodes,
                          is_test_log=True,
                          testing_freq=testing_freq)


def _create_rm_state_logs(n_episodes: int, log_dir: str, last_timestep_info: dict,
                          all_recorded_rm_states: dict, rm_relearned_episodes: dict, is_test_log: bool,
                          testing_freq: int = 1):
    for env_id, test_dicts in last_timestep_info.items():
        x_values = np.arange(1, n_episodes + 1)
        fig_width, fig_height = max(len(x_values) / 8, 6), 6
        plt.figure(f"{env_id}{'_test' if is_test_log else ''}", figsize=(fig_width, fig_height))
        for u in all_recorded_rm_states[env_id]:
            y_values = [u_timestep_dict.get(u, None) for u_timestep_dict in test_dicts]
            y_values = np.array(y_values)

            label = f"u{u}" if isinstance(u, int) else u
            plt.plot(x_values, y_values, label=label, marker='o', markersize=3)

        for relearn_episode in rm_relearned_episodes.get(env_id, []):
            first_ep_with_new_rm = ((relearn_episode - 1) // testing_freq) + 1
            # Moving the line for better readability
            first_ep_with_new_rm -= 0.1
            plt.axvline(x=first_ep_with_new_rm, color='r', linestyle='--')

        plt.xlabel('episode')
        plt.ylabel('last timestep in state u')
        plt.legend()
        img_path = f"{log_dir}/state_transition_{'test' if is_test_log else 'train'}_{env_id}"
        plt.savefig(img_path)
