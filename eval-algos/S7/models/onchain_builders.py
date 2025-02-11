from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import yaml

from utils.allocator import AllocationConfig, allocate_with_constraints
from utils.constants import DEFAULT_CONFIG, DATA_DIR


@dataclass
class DataSnapshot:
    projects_file: str
    metrics_file: str

@dataclass
class SimulationConfig:
    periods: Dict[str, str]
    metrics: Dict[str, float]
    chains: Dict[str, float]
    metric_variants: Dict[str, float]
    normalization: Dict[str, str]

class ScoreCalculator:
    """Handles score computation logic with modular normalization and weighting."""
    def __init__(self, config: SimulationConfig):
        self.config = config
        
    def compute_project_scores(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Main pipeline for computing project scores."""
        analysis = {
            "pivoted_raw_metrics_by_chain": None,
            "pivoted_raw_metrics_weighted_by_chain": None,
            "pivoted_metric_variants": None,
            "normalized_metric_variants": None,
            "weighted_metric_variants": None,
            "aggregated_project_scores": None
        }
        
        analysis["pivoted_raw_metrics_by_chain"] = self._filter_and_pivot_raw_metrics_by_chain(df)
        analysis["pivoted_raw_metrics_weighted_by_chain"] = self._sum_and_weight_raw_metrics_by_chain(analysis["pivoted_raw_metrics_by_chain"])
        analysis["pivoted_metric_variants"] = self._calculate_metric_variants(analysis["pivoted_raw_metrics_weighted_by_chain"])
        analysis["normalized_metric_variants"] = self._normalize_metric_variants(analysis["pivoted_metric_variants"])
        analysis["weighted_metric_variants"] = self._apply_weights_to_metric_variants(analysis["normalized_metric_variants"])
        analysis["aggregated_project_scores"] = self._aggregate_metric_variants(analysis["weighted_metric_variants"])
        # todo: add step to normalize within categories
        
        return analysis

    def _filter_and_pivot_raw_metrics_by_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        metrics_list = list(self.config.metrics.keys())
        periods_list = list(self.config.periods.keys())
        return (
            df.query("metric_name in @metrics_list and measurement_period in @periods_list")
            .pivot_table(
                index=['project_id', 'project_name', 'display_name', 'chain'],
                columns=['measurement_period', 'metric_name'],
                values='amount',
                aggfunc='sum',
                fill_value=0
            )
        )

    def _sum_and_weight_raw_metrics_by_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sum and weight raw metrics by chain."""
        chain_weights = pd.Series(self.config.chains)
        return (
            df
            .mul(df.index.get_level_values('chain').map(chain_weights).fillna(1.0), axis=0)
            .groupby(['project_id', 'project_name', 'display_name']).sum()
        )

    def _calculate_metric_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate adoption, growth, and retention variants for each metric."""
        current_period = next(period for period, label in self.config.periods.items() if label == 'current')
        previous_period = next(period for period, label in self.config.periods.items() if label == 'previous')
        
        variant_scores = {}
        for metric in self.config.metrics.keys():
            current_vals = df[(current_period, metric)].fillna(0)
            prev_vals = df[(previous_period, metric)].fillna(0)

            variant_scores[(metric,'Adoption')] = current_vals
            variant_scores[(metric,'Growth')] = current_vals - prev_vals
            variant_scores[(metric,'Retention')] = pd.concat([current_vals, prev_vals], axis=1).min(axis=1)

        return pd.DataFrame(variant_scores)

    def _normalize_metric_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize variant scores using specified method."""
        df_normalized = df.copy()
        
        for col in df_normalized.columns:
            values = df_normalized[col].values
            if self.config.normalization['method'] == 'minmax':
                df_normalized[col] = self._minmax_scale(values)
            elif self.config.normalization['method'] == 'robust':
                df_normalized[col] = self._robust_scale(values)
                
        return df_normalized

    def _minmax_scale(self, values: np.ndarray) -> np.ndarray:
        """Min-max normalization with handling for edge cases."""
        C = self.config.normalization.get('center_value', 0.5)
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        if max_val == min_val:
            return np.full_like(values, C)
        return np.nan_to_num((values - min_val) / (max_val - min_val), nan=0.0)

    def _robust_scale(self, values: np.ndarray) -> np.ndarray:
        """Robust scaling using IQR with configurable limits."""
        C = self.config.normalization.get('center_value', 0.5)
        lim = self.config.normalization.get('robust_scale_limit', 3)
        
        median = np.nanmedian(values)
        q1 = np.nanpercentile(values, 25)
        q3 = np.nanpercentile(values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return np.full_like(values, C)
            
        scaled = (values - median) / iqr
        clipped = np.clip(scaled, -lim, lim)
        return (clipped + lim) / (2 * lim)

    def _apply_weights_to_metric_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply metric and variant weights to normalized scores."""
        df_weighted = df.copy()
        
        for metric, m_weight in self.config.metrics.items():
            for variant, v_weight in self.config.metric_variants.items():
                weight = m_weight * v_weight
                df_weighted[(metric, variant)] *= weight
                
        return df_weighted

    def _aggregate_metric_variants(self, df: pd.DataFrame, method: str = 'geometric_mean') -> pd.DataFrame:
        """Aggregate metric variants for each project."""
        agg_df = df.copy()
        if method == 'sum':
            agg_df['project_score'] = df.sum(axis=1)
        elif method == 'geometric_mean':
            epsilon = 1e-9
            agg_df['project_score'] = np.exp(np.log(df + epsilon).mean(axis=1))
        else:
            raise ValueError(f"Invalid aggregation method: {method}")
        return agg_df

def load_yaml_configs(config_path: str) -> Tuple[DataSnapshot, SimulationConfig, AllocationConfig]:
    """Load and parse configuration files."""
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    data_snapshot = DataSnapshot(
        projects_file=yaml_config['data_snapshot']['projects_file'],
        metrics_file=yaml_config['data_snapshot']['metrics_file']
    )
    
    sim_config = SimulationConfig(
        periods=yaml_config['simulation']['periods'],
        metrics=yaml_config['simulation']['metrics'],
        chains=yaml_config['simulation']['chains'],
        metric_variants=yaml_config['simulation']['metric_variants'],
        normalization=yaml_config['simulation']['normalization']
    )
    
    alloc_config = AllocationConfig(
        budget=yaml_config['allocation']['budget'],
        min_amount_per_project=yaml_config['allocation']['min_amount_per_project'],
        max_share_per_project=yaml_config['allocation']['max_share_per_project'],
        max_iterations=yaml_config['allocation'].get('max_iterations', 50)
    )
    
    return data_snapshot, sim_config, alloc_config

def load_csv_data(data_snapshot: DataSnapshot) -> pd.DataFrame:
    """Load and preprocess CSV data."""
    filename = lambda pathname: DATA_DIR / pathname
    df_projects = pd.read_csv(filename(data_snapshot.projects_file))
    df_metrics = pd.read_csv(filename(data_snapshot.metrics_file))
    df_metrics['measurement_period'] = pd.to_datetime(df_metrics['sample_date']).dt.strftime('%b %Y')
    return df_metrics.merge(df_projects, on='project_id')

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    def replace_label(col):
        if isinstance(col, str):
            return col
        elif isinstance(col, tuple):
            return ": ".join(col)
        return col
    new_df = df.copy()
    new_df.columns = [replace_label(col) for col in df.columns]
    return new_df

def run_allocation(
    config_path: str = DEFAULT_CONFIG
) -> Dict[str, pd.DataFrame]:
    """
    Run the allocation pipeline and return analysis results and rewards.
    
    Returns:
        Tuple containing:
        - Dictionary of analysis steps (DataFrames from each stage)
        - Series of final reward allocations
    """
    # Load configurations
    data_snapshot, sim_config, alloc_config = load_yaml_configs(config_path)

    # Initialize components and run pipeline
    data = load_csv_data(data_snapshot)
    score_calculator = ScoreCalculator(sim_config)
    
    # Run analysis pipeline
    analysis = score_calculator.compute_project_scores(data)

    # Derive project scores
    scores_df = analysis['aggregated_project_scores']
    scores_series = scores_df['project_score']
    normalized_scores_series = scores_series / scores_series.sum()
    normalized_scores_series.name = 'Weighted Score'
    
    # Allocate rewards
    rewards_series = allocate_with_constraints(normalized_scores_series, alloc_config)
    rewards_series.name = 'OP Rewards'

    # Flatten columns and join with rewards
    df1 = flatten_columns(analysis['pivoted_raw_metrics_weighted_by_chain'])
    df2 = flatten_columns(analysis['pivoted_metric_variants'])
    results_df = (
        df1
        .join(df2)
        .join(normalized_scores_series)
        .join(rewards_series)
        .sort_values(by='OP Rewards', ascending=False)
    )
    analysis['results'] = results_df

    print(f"\nNumber of projects analyzed: {len(results_df)}")
    return analysis

def save_results(analysis: Dict[str, pd.DataFrame], config_path: str = DEFAULT_CONFIG) -> None:
    """Save analysis results to CSV."""
    results_df = analysis['results']
    results_df.to_csv(DATA_DIR / f"{config_path.stem}_results.csv")
    print(f"Results saved to {DATA_DIR / config_path.stem}_results.csv")

def main() -> None:
    analysis = run_allocation()
    save_results(analysis)
    
if __name__ == "__main__":
    main() 