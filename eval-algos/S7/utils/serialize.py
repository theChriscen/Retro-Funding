#!/usr/bin/env python3

import os
import sys
import pandas as pd
from typing import Optional
import argparse


# Add the project root and eval-algos to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
eval_algos_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.extend([project_root, eval_algos_dir])

# Import config
from S7.utils.config import get_measurement_period_paths
from S7.utils.consolidate_rewards import consolidate_rewards, save_consolidated_rewards


def clean_json(dataframe: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a clean JSON string with proper formatting.
    
    Args:
        dataframe: The pandas DataFrame to convert to JSON
        
    Returns:
        A clean JSON string with proper formatting and escaped forward slashes
    """
    json_str = dataframe.to_json(orient='records', indent=2)
    clean_json_str = json_str.replace('\\/', '/')
    return clean_json_str


def serialize_devtooling_results(measurement_period: str, df_rewards: pd.DataFrame) -> None:
    """
    Serialize devtooling metrics and rewards data into a single JSON file.
    
    Args:
        measurement_period: The measurement period (e.g., 'M1', 'M2')
        df_rewards: DataFrame containing consolidated rewards data
    """
    # Get the data and outputs directories from config
    paths = get_measurement_period_paths(measurement_period)
    data_dir = paths['data']
    outputs_dir = paths['outputs']
    
    # Ensure outputs directory exists
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Read devtooling metrics
    devtooling_metrics_path = os.path.join(data_dir, "devtooling__raw_metrics.json")
    df_metrics = pd.read_json(devtooling_metrics_path)

    # Filter rewards for devtooling round and merge
    df_devtooling_rewards = df_rewards[df_rewards['round_id'] == '7']
    df_merged = df_metrics.merge(
        df_devtooling_rewards[['op_atlas_id', 'op_reward', 'round_id']],
        left_on='project_name',
        right_on='op_atlas_id',
        how='outer'
    )

    # Serialize and save results
    clean_json_str = clean_json(df_merged)
    output_path = os.path.join(outputs_dir, "devtooling__results.json")
    with open(output_path, 'w') as file:
        file.write(clean_json_str)
    
    print(f"✓ Saved devtooling results to {output_path}")


def serialize_onchain_results(measurement_period: str, df_rewards: pd.DataFrame) -> None:
    """
    Serialize onchain metrics and rewards data into a single JSON file.
    
    Args:
        measurement_period: The measurement period (e.g., 'M1', 'M2')
        df_rewards: DataFrame containing consolidated rewards data
    """
    # Get the data and outputs directories from config
    paths = get_measurement_period_paths(measurement_period)
    data_dir = paths['data']
    outputs_dir = paths['outputs']
    
    # Ensure outputs directory exists
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Read onchain project metadata
    onchain_metadata_path = os.path.join(data_dir, "onchain__project_metadata.csv")
    df_metrics = pd.read_csv(onchain_metadata_path)
    
    # Filter rewards for onchain round and merge
    df_onchain_rewards = df_rewards[df_rewards['round_id'] == '8']
    df_merged = df_metrics.merge(
        df_onchain_rewards[['op_atlas_id', 'op_reward', 'round_id']],
        left_on='project_name',
        right_on='op_atlas_id',
        how='outer'
    )
    
    # Clean up and prepare data
    df_merged['is_eligible'] = df_merged['is_eligible'].fillna(False)
    
    # Serialize and save results
    clean_json_str = clean_json(df_merged)
    output_path = os.path.join(outputs_dir, "onchain__results.json")
    with open(output_path, 'w') as file:
        file.write(clean_json_str)
    
    print(f"✓ Saved onchain results to {output_path}")


def main():
    """
    Main entry point for the serialization script.
    """
    parser = argparse.ArgumentParser(description='Serialize metrics and rewards data for a specific measurement period')
    parser.add_argument('--measurement-period', '-m', type=str, required=True,
                       help='Measurement period (e.g., M1, M2)')
    
    args = parser.parse_args()
    
    # Consolidate rewards first
    df_rewards = consolidate_rewards(args.measurement_period)
    if df_rewards.empty:
        print("No rewards data found to serialize")
        return
        
    # Save consolidated rewards for reference
    save_consolidated_rewards(df_rewards, args.measurement_period)
    
    # Serialize both devtooling and onchain results
    serialize_devtooling_results(args.measurement_period, df_rewards)
    serialize_onchain_results(args.measurement_period, df_rewards)


if __name__ == "__main__":
    main()
