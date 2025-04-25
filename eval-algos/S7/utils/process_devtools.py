#!/usr/bin/env python3

import pandas as pd
import sys
import argparse
from pathlib import Path
import os
import yaml

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
sys.path.append(project_root)

# Add eval-algos to the Python path
eval_algos_dir = os.path.join(project_root, 'eval-algos')
sys.path.append(eval_algos_dir)

from S7.models.allocator import AllocationConfig, allocate_with_constraints
from S7.models.devtooling_openrank import DevtoolingCalculator, load_config, load_data
from S7.utils.config import (
    get_measurement_period_paths,
    get_model_yaml_path,
    get_output_path,
    ensure_directories
)

def process_scores(measurement_period: str, model_yaml: str):
    """
    Process devtooling scores for a specific measurement period and model.
    
    Args:
        measurement_period: The measurement period (e.g., 'M1', 'M2')
        model_yaml: The model YAML file name without extension
    """
    # Ensure all necessary directories exist
    ensure_directories(measurement_period)

    # Get paths from config
    weights_path = get_model_yaml_path(measurement_period, model_yaml)
    output_path = get_output_path(measurement_period, model_yaml)

    # Load YAML configuration
    with open(weights_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize allocation configuration from YAML
    alloc = AllocationConfig(**config['allocation'])

    # Load model configuration and data
    ds, sim_cfg = load_config(weights_path)
    data = load_data(ds)

    # Run analysis
    calculator = DevtoolingCalculator(sim_cfg)
    analysis = calculator.run_analysis(*data)

    # Extract scores and calculate rewards
    scores = (
        analysis['devtooling_project_results']
        .set_index('project_name')
        ['v_aggregated']
        .sort_values(ascending=False)
    )
    rewards = allocate_with_constraints(scores, alloc, print_results=False)
    rewards.name = 'op_reward'

    # Create combined dataframe with scores and rewards
    df_scores_rewards = pd.concat([scores.to_frame('weighted_score'), rewards], axis=1)
    
    # Save to CSV
    df_scores_rewards.index.name = 'project_id'
    df_scores_rewards.sort_values(by='weighted_score', ascending=False, inplace=True)
    df_scores_rewards.to_csv(output_path, index=True)
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process devtooling scores and calculate rewards')
    parser.add_argument('--measurement-period', '-m', type=str, required=True,
                       help='Measurement period (e.g., M1, M2)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model YAML file name without extension')
    
    args = parser.parse_args()
    process_scores(args.measurement_period, args.model)

if __name__ == "__main__":
    main()
