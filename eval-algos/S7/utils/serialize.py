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


def serialize_devtooling_results(measurement_period: str, df_rewards: Optional[pd.DataFrame] = None) -> None:
    """
    Serialize devtooling metrics and rewards data into a single JSON file.
    
    Args:
        measurement_period: The measurement period (e.g., 'M1', 'M2')
        df_rewards: Optional DataFrame containing consolidated rewards data
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

    # Rename columns
    if 'project_id' in df_metrics.columns:
        df_metrics = df_metrics.rename(columns={'project_id': 'oso_project_id'})
    if 'project_name' in df_metrics.columns:
        df_metrics = df_metrics.rename(columns={'project_name': 'op_atlas_id'})

    # Initialize merged dataframe with metrics
    df_merged = df_metrics.copy()
    df_merged['op_reward'] = None
    df_merged['round_id'] = '7'

    # Merge with rewards if available
    if df_rewards is not None and not df_rewards.empty:
        # Filter rewards for devtooling round and merge
        df_devtooling_rewards = df_rewards[df_rewards['round_id'] == '7']
        if not df_devtooling_rewards.empty:
            # Select only op_reward and round_id from rewards to avoid duplicate op_atlas_id
            df_merged = df_metrics.merge(
                df_devtooling_rewards[['op_atlas_id', 'op_reward', 'round_id']],
                on='op_atlas_id',
                how='outer'
            )
    df_merged['round_id'] = '7'

    # Serialize and save results
    clean_json_str = clean_json(df_merged)
    output_path = os.path.join(outputs_dir, "devtooling__results.json")
    with open(output_path, 'w') as file:
        file.write(clean_json_str)
    
    print(f"✓ Saved devtooling results to {output_path}")


def serialize_onchain_results(measurement_period: str, df_rewards: Optional[pd.DataFrame] = None) -> None:
    """
    Serialize onchain metrics and rewards data into a single JSON file.
    
    Args:
        measurement_period: The measurement period (e.g., 'M1', 'M2')
        df_rewards: Optional DataFrame containing consolidated rewards data
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
    
    # Rename columns
    if 'project_id' in df_metrics.columns:
        df_metrics = df_metrics.rename(columns={'project_id': 'oso_project_id'})
    if 'project_name' in df_metrics.columns:
        df_metrics = df_metrics.rename(columns={'project_name': 'op_atlas_id'})
    
    # Initialize merged dataframe with metrics
    df_merged = df_metrics.copy()
    df_merged['op_reward'] = None
    df_merged['round_id'] = '8'
    
    # Merge with rewards if available
    if df_rewards is not None and not df_rewards.empty:
        # Filter rewards for onchain round and merge
        df_onchain_rewards = df_rewards[df_rewards['round_id'] == '8']
        if not df_onchain_rewards.empty:
            # Select only op_reward and round_id from rewards to avoid duplicate op_atlas_id
            df_merged = df_metrics.merge(
                df_onchain_rewards[['op_atlas_id', 'op_reward', 'round_id']],
                on='op_atlas_id',            
                how='outer'
            )
            # Drop the duplicate op_atlas_id column if it exists
            if 'op_atlas_id_y' in df_merged.columns:
                df_merged = df_merged.drop(columns=['op_atlas_id_y'])
                df_merged = df_merged.rename(columns={'op_atlas_id_x': 'op_atlas_id'})
    
    # Clean up and prepare data
    df_merged['is_eligible'] = df_merged['is_eligible'].fillna(False)
    df_merged['round_id'] = '8'
    
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
    
    # Try to consolidate rewards
    df_rewards = None
    try:
        df_rewards = consolidate_rewards(args.measurement_period)
        if not df_rewards.empty:
            # Save consolidated rewards for reference
            save_consolidated_rewards(df_rewards, args.measurement_period)
        else:
            print("No rewards data found, will serialize with null rewards")
    except Exception as e:
        print(f"Error consolidating rewards: {e}")
        print("Will serialize with null rewards")
    
    # Serialize both devtooling and onchain results
    serialize_devtooling_results(args.measurement_period, df_rewards)
    serialize_onchain_results(args.measurement_period, df_rewards)


if __name__ == "__main__":
    main()
