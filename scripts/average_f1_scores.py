#!/usr/bin/env python3
"""
Script to read TensorBoard logs from a directory and average all metrics starting with "F1-Score".
"""

import argparse
import os
import glob
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from tbparse import SummaryReader


def find_tensorboard_dirs(root_dir: str) -> List[str]:
    """
    Find all directories containing TensorBoard event files.
    TensorBoard logs are typically in directories with .tfevents files.
    """
    tensorboard_dirs = []
    
    # Walk through the directory and find all subdirectories
    for root, dirs, files in os.walk(root_dir):
        # Check if this directory contains tensorboard event files
        event_files = glob.glob(os.path.join(root, "events.out.tfevents.*"))
        if event_files:
            tensorboard_dirs.append(root)
    
    return tensorboard_dirs


def extract_f1_scores(logdir: str, use_last_value: bool = True) -> Dict[str, Union[float, List[float]]]:
    """
    Extract F1-Score metrics from a TensorBoard log directory and aggregate across IDs.

    Tags are expected like: F1-Score/<id>/test_<prop>. We'll group across <id> so that
    F1-Score/0/test_d and F1-Score/1/test_d both contribute to a single key "F1-Score/test_d".

    Args:
        logdir: Directory containing TensorBoard logs
        use_last_value: If True, return only the last value per tag (highest step) and then
                        average across IDs within this logdir. If False, return all values
                        across IDs (concatenated lists) for each grouped metric.

    Returns:
        Dictionary mapping grouped metric names (e.g., "F1-Score/test_d") to either a single
        float (use_last_value=True) or a list of floats (use_last_value=False).
    """
    try:
        reader = SummaryReader(logdir)
        df = reader.scalars

        # Filter for metrics starting with either "F1-Score" or "F1_Score"
        f1_mask = df['tag'].str.startswith('F1-Score', na=False) | df['tag'].str.startswith('F1_Score', na=False)
        f1_metrics = df[f1_mask]

        # Aggregate into group keys: "F1-Score/<suffix>" where suffix is everything after the ID
        grouped_values: Dict[str, List[float]] = {}

        for tag in f1_metrics['tag'].unique():
            parts = str(tag).split('/')
            if len(parts) >= 3:
                # Normalize prefix to "F1-Score" for grouping
                prefix = 'F1-Score'
                suffix = '/'.join(parts[2:])  # e.g., test_d
                group_key = f"{prefix}/{suffix}"
            else:
                # Fallback: keep the tag as-is if it doesn't match expected structure
                group_key = str(tag)

            tag_data = f1_metrics[f1_metrics['tag'] == tag]
            if use_last_value:
                last_row = tag_data.loc[tag_data['step'].idxmax()]
                grouped_values.setdefault(group_key, []).append(float(last_row['value']))
            else:
                grouped_values.setdefault(group_key, []).extend(tag_data['value'].astype(float).tolist())

        # If using last values, average across IDs within this directory for each group
        if use_last_value:
            return {k: float(np.mean(v)) for k, v in grouped_values.items() if len(v) > 0}
        else:
            return grouped_values
    except Exception as e:
        print(f"Warning: Could not read {logdir}: {e}")
        return {}


def average_f1_scores(root_dir: str, use_last_value: bool = True) -> Dict[str, float]:
    """
    Read all TensorBoard logs from a directory and average F1-Score metrics.
    
    Args:
        root_dir: Root directory containing TensorBoard logs
        use_last_value: If True, average the final value per metric per directory.
                       If False, average all logged values.
    
    Returns:
        Dictionary mapping metric names to their average values
    """
    # Find all tensorboard directories
    logdirs = find_tensorboard_dirs(root_dir)
    
    if not logdirs:
        print(f"No TensorBoard log directories found in {root_dir}")
        return {}
    
    print(f"Found {len(logdirs)} TensorBoard log directories")
    if use_last_value:
        print("Using last value per metric per directory (final values)")
    else:
        print("Using all logged values")
    
    # Collect F1-Score metrics from all directories
    all_f1_scores: Dict[str, List[float]] = {}
    
    for logdir in logdirs:
        print(f"Reading: {logdir}")
        f1_scores = extract_f1_scores(logdir, use_last_value=use_last_value)
        
        for tag, value in f1_scores.items():
            if tag not in all_f1_scores:
                all_f1_scores[tag] = []
            
            if use_last_value:
                # value is a single float
                all_f1_scores[tag].append(value)
            else:
                # value is a list of floats
                all_f1_scores[tag].extend(value)
    
    # Calculate averages
    averages = {}
    for tag, values in all_f1_scores.items():
        if values:
            averages[tag] = np.mean(values)
            print(f"\n{tag}:")
            print(f"  Number of {'runs' if use_last_value else 'values'}: {len(values)}")
            print(f"  Average: {averages[tag]:.6f}")
            print(f"  Min: {np.min(values):.6f}")
            print(f"  Max: {np.max(values):.6f}")
            print(f"  Std: {np.std(values):.6f}")
    
    return averages


def main():
    parser = argparse.ArgumentParser(
        description="Average F1-Score metrics from TensorBoard logs in a directory"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing TensorBoard logs"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output file to save results (CSV format)"
    )
    parser.add_argument(
        "--all-values",
        action="store_true",
        help="Average all logged values instead of just the final value per run"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    print(f"Reading TensorBoard logs from: {args.directory}\n")
    
    use_last_value = not args.all_values
    averages = average_f1_scores(args.directory, use_last_value=use_last_value)
    
    if not averages:
        print("\nNo F1-Score metrics found in the TensorBoard logs.")
        return
    
    # Save to file if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Average'])
            for tag, avg in sorted(averages.items()):
                writer.writerow([tag, avg])
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

