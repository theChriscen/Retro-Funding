#!/usr/bin/env python3

import pandas as pd
import sys
import argparse
from pathlib import Path
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.abspath(os.path.join(script_dir, '../S7/models'))
sys.path.append(models_path)

from utils.allocator import AllocationConfig, allocate_with_constraints
from devtooling_openrank import DevtoolingCalculator, load_config, load_data

def process_scores(model_yaml):
    # Initialize allocation configuration
    alloc = AllocationConfig(
        budget=8_000_000/6,
        min_amount_per_project=200, 
        max_share_per_project=0.05
    )

    # Load configuration and data using absolute paths
    config_path = os.path.join(os.path.dirname(models_path), 'weights', f'{model_yaml}.yaml')
    ds, sim_cfg = load_config(config_path)
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
    
    # Construct output path based on model name
    model_parts = model_yaml.split('_')
    milestone = "M1"  # Always use M1 for this model
    category = "devtooling"
    output_path = f'eval-algos/S7/data/{milestone}/{category}/{model_yaml}_rewards.csv'
    
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df_scores_rewards.index.name = 'project_id'
    df_scores_rewards.sort_values(by='weighted_score', ascending=False, inplace=True)
    df_scores_rewards.to_csv(output_path, index=True)
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process scores and calculate rewards based on model YAML')
    parser.add_argument('model_yaml', type=str, 
                       help='Name of the model YAML file (without .yaml extension)')
    
    args = parser.parse_args()
    process_scores(args.model_yaml)

if __name__ == "__main__":
    main()
