import re
import glob
import os.path
import time
import sys

import pandas as pd
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

    # Convert -1 rewards to 0, as for the final results we don't care about the difference between timeouts and failures
    reward_data.loc[reward_data.value < 0, 'value'] = 0

    return epsilon_data, reward_data, num_steps_data


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


if __name__ == "__main__":
    dir_name = sys.argv[1]

    log_dir_reg = f'logs/{dir_name}/*'
    preprocessed_log_dir = f'logs/preprocessed/{dir_name}'

    # Match any file name that start with some text and end with _{seed}_{sensor_confidence}
    file_pattern = r"(.*)_(?P<seed>\d+)_(?P<sensor_confidence>[\d.]+)"

    log_dirs = glob.glob(log_dir_reg)
    print(f"[{time.asctime()}]: Started shrinking {len(log_dirs)} Tensorboard log files")

    for i, log_dir in enumerate(log_dirs):

        print(f"[{time.asctime()}][{i}/{len(log_dirs)}]: Starting to shrink: {log_dir}")

        df = load_logged_data(log_dir, file_pattern)
        eps, rew, steps = preprocess_results(df)

        run_session = os.path.basename(log_dir)
        out_dir = os.path.join(preprocessed_log_dir, run_session)
        write_preprocess_data(out_dir, eps, rew, steps)