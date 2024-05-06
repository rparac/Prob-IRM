import argparse
import os
import re
import glob
import time

import pandas as pd
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter

# Important - should include at least every seed
# log_dir_reg = '/gpfs/home/rp218/rm-marl/saved_logs/*'
# Match any file name that start with some text and end with _{seed}_{sensor_confidence}
file_pattern = r"(.*)_(?P<seed>\d+)_(?P<sensor_confidence>[\d.]+)"


def to_dataframe(base_dir, log_dirs, extraction_pattern):
    dfs = []
    for logdir in log_dirs:
        # inside the directory with timings. Take the most recent one
        # dir_under_consideration = max(os.listdir(logdir))

        reader = SummaryReader(f"{base_dir}/{logdir}")
        df = reader.scalars
        match = re.match(extraction_pattern, logdir)
        df['logdir'] = logdir
        df['seed'] = int(match.group("seed"))
        df['sensor_confidence'] = float(match.group("sensor_confidence"))
        dfs.append(df)
    return pd.concat(dfs)


def aggregate_by_tag(df):
    return df.groupby(['tag', 'step', 'sensor_confidence']).agg({'value': ['mean', 'std']}).reset_index()


def aggregate_by_env(df):
    df['tag_without_env'] = df['tag'].str.rsplit('/', n=1).str[0]
    return df.groupby(['tag_without_env', 'step', 'sensor_confidence']).agg({'value': ['mean', 'std']}).reset_index()


def write_tagged_results(grouped_df, tag_column_name, output_dir):
    writer = SummaryWriter(output_dir)
    # tags - eval/success_rate/* eval/failure_rate/*
    for tag in grouped_df[tag_column_name].unique():
        val = grouped_df.loc[grouped_df[tag_column_name] == tag]
        for index, row in val.iterrows():
            curr_step = row["step"].iloc[0]
            writer.add_scalar(f"{tag}/mean", row['value']['mean'], curr_step)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', help="Directory where directories with experiments are stored")
    parser.add_argument('output', default='summary')
    args = parser.parse_args()

    log_dirs = os.listdir(args.basedir)

    all_df = to_dataframe(args.basedir, log_dirs, file_pattern)
    print("Generated pandas dataframe")
    grouped_result = aggregate_by_tag(all_df)
    write_tagged_results(grouped_result, tag_column_name="tag", output_dir=f"{args.basedir}/{args.output}")
    print("First results are written")

    env_grouped_results = aggregate_by_env(all_df)
    write_tagged_results(env_grouped_results, tag_column_name="tag_without_env",
                         output_dir=f"{args.basedir}/{args.output}")
    print(f"Second results are written")
