import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sympy import symbols, solve

import os
import json
import os.path
import functools

BASE_COLUMN_NAMES = {
    'training_iteration',
    'time_total_s',
    'env_runners/original_episode_return_mean',
    'experiment',
    'run_group_id',
    'noise_level',
    'seed'
}


def parse_json_into_df(json_file_path):

    relevant_dict_keys = ['training_iteration', 'time_total_s']

    results_dict = {
        key: []
        for key in relevant_dict_keys
    }

    with open(json_file_path) as json_file:

        for i, line in enumerate(json_file):
            json_data = json.loads(line)

            for key in relevant_dict_keys:
                results_dict[key].append(json_data[key])

            if 'original_episode_return_mean' in json_data['env_runners']:
                try:
                    results_dict['env_runners/original_episode_return_mean'].append(json_data['env_runners']['original_episode_return_mean'])
                except KeyError:
                    results_dict['env_runners/original_episode_return_mean'] = [json_data['env_runners']['original_episode_return_mean']]

            else:
                for subkey in json_data['env_runners']['agent_episode_returns_mean']:
                    try:
                        results_dict[f'env_runners/agent_episode_returns_mean/{subkey}'].append(json_data['env_runners']['agent_episode_returns_mean'][subkey])
                    except KeyError:
                        results_dict[f'env_runners/agent_episode_returns_mean/{subkey}'] = [json_data['env_runners']['agent_episode_returns_mean'][subkey]]

    return pd.DataFrame(results_dict)


def load_results_data(results_base_dir: str, experiment: str, results_format: str = 'csv') -> pd.DataFrame:

    assert results_format in ['csv', 'json'], f'Requested unsupported results format: {results_format}'

    if results_format == 'csv':
        single_results_filename = 'progress.csv'
        results_to_df = pd.read_csv
    elif results_format == 'json':
        single_results_filename = 'result.json'
        # results_to_df = functools.partial(pd.read_json, orient='records')
        results_to_df = parse_json_into_df
    else:
        single_results_filename = ''
        assert False

    results_data = []

    print(f'[.] Loading {results_format.upper()} results for experiment: {experiment}')

    run_path = experiment.rsplit('/', maxsplit=1)[0]
    run_group_pattern = experiment.split('/')[-1]

    exp_results_path = os.path.join(results_base_dir, run_path)

    subdirs = [d for d in os.scandir(exp_results_path) if d.is_dir() and d.name.startswith(run_group_pattern)]
    print(f'\t -> Found {len(subdirs)} runs')
    print()

    for subdir in subdirs:

        # Locate the file containing the results and parse them
        results_file_fullpath = None
        for dirpath, _, filenames in os.walk(subdir.path):
            if single_results_filename in filenames:
                results_file_fullpath = os.path.join(dirpath, single_results_filename)
                break

        if results_file_fullpath is None:
            print(f'[WARNING] Could not find {results_format.upper()} results for run: {subdir.name}')
            continue

        else:

            run_results_df = results_to_df(results_file_fullpath)

            # Experiments without reward shaping do not have this column
            if 'env_runners/original_episode_return_mean' not in run_results_df.columns.values:

                agent_returns_columns = [
                    c for c in run_results_df.columns.values
                    if c.startswith('env_runners/agent_episode_returns_mean/')
                ]
                agent_returns_df = run_results_df[agent_returns_columns]
                run_results_df['env_runners/original_episode_return_mean'] = agent_returns_df.mean(axis=1)

            # Keep only relevant columns
            kept_columns = {
                'training_iteration',
                'time_total_s',
                'env_runners/original_episode_return_mean',
                # 'env_runners/num_episodes',
            }
            all_columns = set(run_results_df.columns.values)
            droppable_columns = list(all_columns - kept_columns)
            run_results_df = run_results_df.drop(columns=droppable_columns)

            # Read common run parameters
            run_name_components = subdir.name.replace(run_group_pattern + '_', '').split("_")
            run_results_df['experiment'] = run_path
            run_results_df['run_group_id'] = subdir.name.rsplit('_', maxsplit=2)[0]
            run_results_df['noise_level'] = run_name_components[-2]
            run_results_df['seed'] = run_name_components[-1]

            # Read additional run parameters
            additional_parameters = run_name_components[0:-2]
            for param in additional_parameters:
                param_components = param.split("-")
                param_name = param_components[0]
                param_value = param_components[1]

                run_results_df[param_name] = param_value

            # Extend to N
            extend_to = 500
            last_row = run_results_df.iloc[[-1]]
            num_iters = last_row.iloc[0]['training_iteration']
            if num_iters < extend_to:
                new_rows = pd.concat([last_row] * (extend_to - num_iters), ignore_index=True)
                run_results_df = pd.concat([run_results_df, new_rows], ignore_index=True)
                run_results_df['training_iteration'] = range(1, len(run_results_df)+1)

            # Get first 10000 dataframes
            run_results_df = run_results_df[:10000]

            results_data.append(run_results_df)



    results_df = pd.concat(results_data)
    results_df = results_df.reset_index(drop=True)

    return results_df


def average_data_over_groups(results_df: pd.DataFrame) -> pd.DataFrame:

    all_columns = set(results_df.columns.values)
    extra_param_columns = all_columns - BASE_COLUMN_NAMES

    groups = results_df.groupby([
        'experiment',
        'run_group_id',
        'noise_level',
        'training_iteration',
        *extra_param_columns
    ], dropna=False)  # dropna=False required to avoid ignoring index values with NaNs in them

    averages_df = groups.agg({
        'env_runners/original_episode_return_mean': [
            'mean',
            lambda x: np.percentile(x, q=25),
            lambda x: np.percentile(x, q=75)
        ],
        'time_total_s': [
            'mean',
            lambda x: np.percentile(x, q=25),
            lambda x: np.percentile(x, q=75)
        ]
    }).rename(level=0, columns={
        'env_runners/original_episode_return_mean': 'ep_return',
        'time_total_s': 'runtime'
    }).rename(level=1, columns={
        '<lambda_0>': '25-perc',
        '<lambda_1>': '75-perc'
    }).reset_index(level=3, drop=False)  # Extract training iteration number from index to column

    return averages_df


def sensitivity_from_posterior(posterior, prior=2/108):

    s = symbols('s')
    f = posterior - (s * prior) / (s * prior + (1 - s) * (1 - prior))
    return solve(f, s)[0]


def posterior_from_sensitivity(sensitivity, prior=2/108):

    p = symbols('p')
    f = p + sensitivity + prior  # TODO
    return solve(f, p)[0]


def plot_learning_curves(results_df: pd.DataFrame):

    averages_df = average_data_over_groups(results_df)
    experiments = set(averages_df.index.get_level_values('experiment'))
    run_groups = set(averages_df.index.get_level_values('run_group_id'))
    noise_levels = set(averages_df.index.get_level_values('noise_level'))

    line_colors = ('royalblue', 'forestgreen', 'orange', 'red')

    assert len(noise_levels) <= len(line_colors), 'Noise levels > Defined colors'

    for exp in experiments:

        exp_averages_df = averages_df.loc[exp]

        # Determine which run groups are relevant for the current experiment
        exp_run_groups = []
        for group_id in run_groups:

            try:
                _ = exp_averages_df.loc[group_id]
                exp_run_groups.append(group_id)
            except KeyError:
                continue

        for i, group_id in enumerate(exp_run_groups):

            plt.figure(figsize=(5, 5), dpi=150)
            # plt.suptitle(f'Experiment: {exp}')
            title = f'{group_id}'
            # title = title.split("_")[-1]
            # title = "_".join(title)
            # title = "example penalty " + title
            title = title.replace("rg_b", "rg-b")
            # title = title.replace("_", " ")
            title = title.replace("rebalance_classes_", "")
            title = title.replace("-", " = ")
            title = title.replace("rs_deliver_coffee_shaping", "use_reward_shaping")
            # title = "No Shaping"
            plt.title(title)

            group_averages_df = exp_averages_df.loc[group_id]

            # Determine which noise levels are relevant for the current run group
            group_noise_levels = []
            for noise_level in noise_levels:

                try:
                    _ = group_averages_df.loc[noise_level]
                    group_noise_levels.append(noise_level)
                except KeyError:
                    continue

            for j, noise_level in enumerate(sorted(group_noise_levels, reverse=True)):

                noise_level_averages_df = group_averages_df.loc[noise_level]

                x_axis = noise_level_averages_df['training_iteration']
                y_axis = noise_level_averages_df['ep_return']['mean']
                y_shade_high = noise_level_averages_df['ep_return']['75-perc']
                y_shade_low = noise_level_averages_df['ep_return']['25-perc']

                plt.gca().fill_between(
                    x_axis,
                    y_shade_low,
                    y_shade_high,
                    color=line_colors[j],
                    alpha=0.2
                )

                # Quick and dirty dashed line
                if False and noise_level == '0.8':
                    plt.plot(x_axis, y_axis,
                            linewidth=1.5,
                            color=line_colors[j],
                            linestyle='--',
                            label="None",)
                            # label=f'{float(noise_level):.1f}')
                            #  label=f'{float(noise_level):.2f}')
                            #  label=f'{float(noise_level):.5f}')
                else:
                    plt.plot(x_axis, y_axis,
                            linewidth=1.5,
                            color=line_colors[j],
                            label=f'{float(noise_level):.1f}')
                            #  label=f'{float(noise_level):.2f}')
                            #  label=f'{float(noise_level):.5f}')

                plt.ylabel(f'Average return')
                plt.xlabel('Iterations')
                plt.ylim(-0.05, 1.05)
                plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

                plt.legend(title='Sensor Confidence', loc='lower right')
                # plt.legend(title='Threshold', loc='lower right')
                plt.grid(linewidth=0.3)
                plt.tight_layout()

            os.makedirs(f'plots/{exp}', exist_ok=True)
            plt.savefig(f'plots/{exp}/{group_id}.png')
            plt.close()


if __name__ == '__main__':

    csv_results = [
        # # Ablations
        # 'ablations/reward_shaping/rs_deliver_coffee',
    ]


    json_results = [
        # Ablations
        # 'ablations/cross_entropy_threshold/cross_entorpy',
        'ablations/reward_shaping/rs_deliver_coffee',
        # 'ablations/example_penalty_multiplier/ex_penalty_multiplier',

        # "ablations/noise_all/all_deliver_coffee_rm_learning",
        # "ablations/noise_all/all_deliver_coffee_perfect_rm",

        # "ablations/rebalance_classes/rebalance_classes",
        # "ablations/thresholding/threshold",

        # 'probirm/office/deliver_coffee_rm_learning',
        # 'probirm/office/coffee_mail_rm_learning',
        # 'probirm/office/visit_abcd_a_rm_learning',

        # 'baselines/office/perfect_rm/deliver_coffee_perfect_rm',
        # 'baselines/office/perfect_rm/coffee_mail_perfect_rm',
        # 'baselines/office/perfect_rm/visit_abcd_a_perfect_rm',

        # 'probirm/water/waterworld_rgb_unrestricted_rm_learning',
        # 'probirm/water/waterworld_rg_b_unrestricted_rm_learning',
        # 'probirm/water/waterworld_rgbc_unrestricted_rm_learning',

        # 'baselines/water/perfect_rm/waterworld_rgb_unrestricted_perfect_rm',
        # 'baselines/water/perfect_rm/waterworld_rg_b_unrestricted_perfect_rm',
        # 'baselines/water/perfect_rm/waterworld_rgbc_unrestricted_perfect_rm',

        # 'baselines/office/perfect_rm/deliver_coffee_perfect_rm',
        # 'baselines/office/perfect_rm/coffee_mail_perfect_rm',
        # 'baselines/office/perfect_rm/visit_abcd_a_perfect_rm',

        # "baselines/office/recurrent/recurrent_deliver_coffee_rm_learning",
        # "baselines/office/recurrent/recurrent_coffee_mail_rm_learning",
        # "baselines/office/recurrent/recurrent_visit_abcd_a_rm_learning",

        # "baselines/water/recurrent/recurrent_waterworld_rgb_rm_learning",
        # "baselines/water/recurrent/recurrent_waterworld_rg_b_rm_learning",
        # "baselines/water/recurrent/recurrent_waterworld_rgbc_rm_learning",
    ]


    # csv_results = [

    #     # Ablations
    #     'ablations/noisy_all/all_deliver_coffee',
    #     'ablations/reward_shaping/rs_deliver_coffee',

    #     # Hand-crafted RM baselines
    #     'baselines/perfect_rm/office/deliver_coffee_perfect_rm',
    #     'baselines/perfect_rm/office/coffee_mail_perfect_rm',
    #     'baselines/perfect_rm/office/visit_abcd_a_perfect_rm',
    #     'baselines/perfect_rm/water/waterworld_rg_b_unrestricted_perfect_rm',
    #     'baselines/perfect_rm/water/waterworld_rgb_unrestricted_perfect_rm',
    #     'baselines/perfect_rm/water/waterworld_RGBc_unrestricted_perfect_rm',

    #     # Recurrent baselines
    #     'baselines/recurrent/office/recurrent_deliver_coffee',
    #     'baselines/recurrent/office/recurrent_coffee_mail',
    #     'baselines/recurrent/office/recurrent_visit_abcd_a',

    #     # Prob-IRM results
    #     'probirm/office/deliver_coffee_rm_learning',
    #     'probirm/office/coffee_mail_rm_learning',
    #     'probirm/office/visit_abcd_a_rm_learning',
    #     'probirm/water/waterworld_rg_b_unrestricted_rm_learning',
    #     'probirm/water/waterworld_rgb_unrestricted_rm_learning',
    #     'probirm/water/waterworld_RGBc_unrestricted_rm_learning',

    # ]

    # json_results = [

    #     'baselines/recurrent/water/recurrent_waterworld_rg_b',
    #     'baselines/recurrent/water/recurrent_waterworld_rgb',
    #     'baselines/recurrent/water/recurrent_waterworld_RGBc',

    # ]

    for exp in csv_results:
        results = load_results_data('results/csv', exp, results_format='csv')
        plot_learning_curves(results)
    json_folder = '/home/rp218/from_hx1'
    for exp in json_results:
        results = load_results_data(json_folder, exp, results_format='json')
        plot_learning_curves(results)
