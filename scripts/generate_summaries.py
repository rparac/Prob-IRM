import re
import glob
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


pd.options.mode.copy_on_write = True


def load_logged_data(logdir, extraction_pattern):
    reader = SummaryReader(logdir)
    match = re.match(extraction_pattern, logdir)

    df = reader.scalars
    df['logdir'] = logdir
    df['seed'] = int(match.group("seed"))
    df['sensor_confidence'] = float(match.group("sensor_confidence"))

    return df


def preprocess_results(df):
    epsilon_data = df[df.tag.str.contains(r'algo/.*/epsilon')]
    reward_data = df[df.tag.str.contains(r'.*/reward/.*')]
    num_steps_data = df[df.tag.str.contains(r'.*/num_steps/.*')]

    # Keep only rows where epsilon = 1, as they allow to derive the episodes where a new RM was learnt
    droppable = epsilon_data[epsilon_data.value != 1.0].index
    epsilon_data = epsilon_data.drop(droppable)

    # Convert -1 rewards to 0, as for the final results we do not care about the difference between timeouts and failures
    reward_data.loc[reward_data.value < 0, 'value'] = 0

    return epsilon_data, reward_data, num_steps_data


def separate_and_concatenate(all_dfs):
    all_ep_rewards, all_ep_steps = [], []
    for df in all_dfs:
        eps, rew, steps = preprocess_results(df)
        ep_rewards, ep_steps = agent_performance_metrics(rew, steps)

        all_ep_rewards.append(ep_rewards)
        all_ep_steps.append(ep_steps)

    ep_rewards = pd.concat(all_ep_rewards)
    ep_steps = pd.concat(all_ep_steps)

    return ep_rewards, ep_steps


def write_preprocess_data(out_dir, epsilon_df, reward_df, num_steps_df):
    writer = SummaryWriter(out_dir)

    for index, row in epsilon_df.iterrows():
        curr_step = row["step"]
        writer.add_scalar(row["tag"], row['value'], curr_step)

    for index, row in reward_df.iterrows():
        curr_step = row["step"]
        writer.add_scalar(row["tag"], row['value'], curr_step)

    for index, row in num_steps_df.iterrows():
        curr_step = row["step"]
        writer.add_scalar(row["tag"], row['value'], curr_step)


def convert_tag_to_env(df):
    df['env'] = df['tag'].str.rsplit('/', n=1).str[1]
    df = df.drop(columns='tag')

    return df


def agent_performance_metrics(reward_df, num_steps_df):
    droppable_rewards = reward_df[reward_df.tag.str.contains(r'eval')].index
    episode_rewards = reward_df.drop(droppable_rewards)
    episode_rewards = convert_tag_to_env(episode_rewards)
    episode_rewards = episode_rewards.rename(columns={'step': 'episode'})
    episode_rewards = episode_rewards.set_index(['logdir', 'env', 'episode'])

    droppable_steps = num_steps_df[num_steps_df.tag.str.contains(f'eval')].index
    episode_steps = num_steps_df.drop(droppable_steps)
    episode_steps = convert_tag_to_env(episode_steps)
    episode_steps = episode_steps.rename(columns={'step': 'episode'})
    episode_steps = episode_steps.set_index(['logdir', 'env', 'episode'])

    return episode_rewards, episode_steps


def aggregate_rewards(df):
    df = df.reset_index()
    df = df.groupby(['sensor_confidence', 'episode']).agg({
        'value': [
            'mean',
            'std',
            'median',
            lambda x: np.percentile(x, q=25),
            lambda x: np.percentile(x, q=75)

        ]
    })

    df = df.rename(level=1, columns={
        '<lambda_0>': '25-perc',
        '<lambda_1>': '75-perc'
    })

    return df


def aggregate_num_steps(df):
    df = df.reset_index()
    df = df.groupby(['sensor_confidence', 'episode']).agg({
        'value': [
            'mean',
            'std',
            'median',
            lambda x: np.percentile(x, q=25),
            lambda x: np.percentile(x, q=75)
        ]
    })

    df = df.rename(level=1, columns={
        '<lambda_0>': '25-perc',
        '<lambda_1>': '75-perc'
    })

    return df


def plot_mean_episodic_metric(agg_metric_df, *,
                              yaxis='',
                              smooth=False,
                              filename=None,
                              posteriors=(0.5, 0.8, 0.9, 1.0),
                              line_colors=('red', 'orange', 'forestgreen', 'royalblue')):
    agg_metric_df = agg_metric_df.reset_index().set_index('sensor_confidence')

    plt.figure(figsize=(6, 5), dpi=150)

    sensor_confidences = list(agg_metric_df.index.unique().sort_values().array)

    for index in agg_metric_df.index.unique():

        episodes = agg_metric_df.loc[index, 'episode']
        mean_rewards = agg_metric_df.loc[index, ('value', 'mean')]

        line_color = line_colors[sensor_confidences.index(index)]
        plot_label = posteriors[sensor_confidences.index(index)]

        if smooth:
            window_size = 100

            # Compute stddev of values over smoothing window to plot shaded area to represent variability
            windowed_stddev = mean_rewards.rolling(window=window_size).std()
            mean_rewards = mean_rewards.rolling(window=window_size).mean()

            shade_upper = mean_rewards + windowed_stddev
            shade_lower = mean_rewards - windowed_stddev

            plt.gca().fill_between(
                episodes,
                shade_lower,
                shade_upper,
                color=line_color,
                alpha=0.2

            )

        plt.plot(episodes, mean_rewards,
                 linewidth='0.5' if not smooth else 1.5,
                 label=plot_label,
                 color=line_color
                 )

        plt.ylabel(f'Average {yaxis}')
        plt.xlabel('Episodes')
        plt.ylim(-0.05, 1.05)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.legend(title='Posterior')
    plt.grid(linewidth=0.3)

    if filename is not None:
        plt.savefig(filename)


def plot_thresholding_results(agg_metric_t07, agg_metric_t09, *,
                              smooth=True):
    agg_metric_t07 = agg_metric_t07.reset_index()
    agg_metric_t09 = agg_metric_t09.reset_index()

    plt.figure(figsize=(6, 5), dpi=150)

    # Plot lower threshold ie: the one that, in theory, should work
    episodes = agg_metric_t07['episode']
    mean_rewards = agg_metric_t07[('value', 'mean')]

    if smooth:
        window_size = 100

        # Compute stddev of values over smoothing window to plot shaded area to represent variability
        windowed_stddev = mean_rewards.rolling(window=window_size).std()
        mean_rewards = mean_rewards.rolling(window=window_size).mean()

        shade_upper = mean_rewards + windowed_stddev
        shade_lower = mean_rewards - windowed_stddev

        plt.gca().fill_between(
            episodes,
            shade_lower,
            shade_upper,
            color="forestgreen",
            alpha=0.2

        )

    plt.plot(episodes, mean_rewards,
             linewidth='0.5' if not smooth else 1.5,
             label="0.7",
             color="forestgreen"
             )

    # Plot higher threshold ie: the one that, in theory, should not work
    episodes = agg_metric_t09['episode']
    mean_rewards = agg_metric_t09[('value', 'mean')]

    if smooth:
        window_size = 100

        # Compute stddev of values over smoothing window to plot shaded area to represent variability
        windowed_stddev = mean_rewards.rolling(window=window_size).std()
        mean_rewards = mean_rewards.rolling(window=window_size).mean()

        shade_upper = mean_rewards + windowed_stddev
        shade_lower = mean_rewards - windowed_stddev

        plt.gca().fill_between(
            episodes,
            shade_lower,
            shade_upper,
            color="red",
            alpha=0.2

        )

    plt.plot(episodes, mean_rewards,
             linewidth='0.5' if not smooth else 1.5,
             label="0.9",
             color="red"
             )

    plt.ylabel(f'Average reward')
    plt.xlabel('Episodes')
    plt.ylim(-0.05, 1.05)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.legend(title='Threshold')
    plt.grid(linewidth=0.3)

    plt.savefig('../results/plots/thresholding_comparison.png')


if __name__ == '__main__':

    file_pattern = r"(.*)_(?P<seed>\d+)_(?P<sensor_confidence>[\d.]+)"

    main_sessions = [
        'with_rm_deliver_coffee',
        'deliver_coffee',
        'with_rm_coffee_mail',
        'coffee_mail',
        'with-rm_visit_abcd_a',
        'visit_abcd_a',
        'all_deliver_coffee'
    ]

    # Baseline + Multiple noise sources results
    for session in main_sessions:

        print(f'Generating plots for session: "{session}"')

        log_dirs = glob.glob(f'../results/preprocessed/{session}/*')
        all_dfs = [load_logged_data(logdir, file_pattern) for logdir in log_dirs]

        ep_rewards, _ = separate_and_concatenate(all_dfs)
        agg_ep_rewards = aggregate_rewards(ep_rewards)

        plot_mean_episodic_metric(
            agg_ep_rewards,
            yaxis='reward',
            smooth=True,
            filename=f'../results/plots/{session}_rewards.png',
            posteriors=(0.5, 0.8, 0.9, 1.0) if session != 'all_deliver_coffee' else (0.75, 0.8, 0.9, 1.0),
            line_colors=('orangered', 'orange', 'forestgreen', 'royalblue')
        )

    # Thresholding results
    print(f'Generating plots for thresholding experiments')

    t7_log_dirs = glob.glob(f'../results/preprocessed/all_t-0.7_deliver_coffee/*')
    t9_log_dirs = glob.glob(f'../results/preprocessed/all_t-0.9_deliver_coffee/*')

    t7_all_dfs = [load_logged_data(logdir, file_pattern) for logdir in t7_log_dirs]
    t9_all_dfs = [load_logged_data(logdir, file_pattern) for logdir in t9_log_dirs]

    t7_ep_rewards, _ = separate_and_concatenate(t7_all_dfs)
    t9_ep_rewards, _ = separate_and_concatenate(t9_all_dfs)

    t7_agg_ep_rewards = aggregate_rewards(t7_ep_rewards)
    t9_agg_ep_rewards = aggregate_rewards(t9_ep_rewards)

    plot_thresholding_results(t7_agg_ep_rewards, t9_agg_ep_rewards)
