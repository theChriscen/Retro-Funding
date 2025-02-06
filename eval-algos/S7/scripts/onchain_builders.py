import pandas as pd
from dataclasses import dataclass
import yaml
from typing import Dict, Any, Tuple

from allocator import AllocationConfig, allocate_with_constraints
from constants import DEFAULT_CONFIG, DATA_DIR


@dataclass
class SimulationConfig:
    projects_file: str
    metrics_file: str
    periods: dict
    metrics: dict
    chains: dict
    metric_variants: dict

    @classmethod
    def from_yaml(cls, yaml_config: dict) -> 'SimulationConfig':
        sim_config = yaml_config['simulation']
        return cls(
            projects_file=DATA_DIR / sim_config['projects_file'],
            metrics_file=DATA_DIR / sim_config['metrics_file'],
            periods=sim_config['periods'],
            metrics=sim_config['metrics'],
            chains=sim_config['chains'],
            metric_variants=sim_config['metric_variants'],
        )


class DataLoader:
    """Handles loading and preprocessing of data from local files."""
    def __init__(self, config: SimulationConfig):
        self.config = config
        
    def load_data(self) -> pd.DataFrame:
        df_projects = pd.read_csv(self.config.projects_file)
        df_metrics = pd.read_csv(self.config.metrics_file)
        df_metrics['measurement_period'] = pd.to_datetime(df_metrics['sample_date']).dt.strftime('%b %Y')
        return df_metrics.merge(df_projects, on='project_id')
        
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        metrics_list = list(self.config.metrics.keys())
        periods_list = list(self.config.periods.keys())
        return (
            df.query("metric_name in @metrics_list and measurement_period in @periods_list")
            .dropna()
            .pivot_table(
                index=['project_id', 'project_name', 'display_name', 'chain'],
                columns=['measurement_period', 'metric_name'],
                values='amount'
            )
        )


class ScoreCalculator:
    """Handles score computation logic."""
    def __init__(self, config: SimulationConfig):
        self.config = config
        
    def compute_project_scores(self, df: pd.DataFrame) -> pd.Series:
        """Compute normalized scores for all projects."""
        df_scores = df.copy()
        df_scores['score'] = df_scores.apply(self._compute_row_score, axis=1)
        
        project_scores = df_scores['score'].groupby(level='project_name').sum()
        total_score = project_scores.sum()
        
        if total_score > 0:
            return project_scores / total_score
        return pd.Series(0, index=project_scores.index)
    
    def _compute_row_score(self, row: pd.Series) -> float:
        """Compute score for a single project-chain row."""
        chain = row.name[3]  # chain is at index 3 after pivot
        chain_weight = self.config.chains.get(chain, 0.0)
        
        current_period = next(period for period, label in self.config.periods.items() if label == 'current')
        previous_period = next(period for period, label in self.config.periods.items() if label == 'previous')
        
        total_metric_score = sum(
            self._compute_metric_score(metric, metric_weight, row, current_period, previous_period)
            for metric, metric_weight in self.config.metrics.items()
        )
        
        return chain_weight * total_metric_score
    
    def _compute_metric_score(
        self,
        metric: str,
        weight: float,
        row: pd.Series, 
        current_period: str,
        previous_period: str
        ) -> float:
        """Compute score for a single metric."""
        current_val = row.get((current_period, metric), 0.0)
        previous_val = row.get((previous_period, metric), 0.0)

        adoption = current_val
        growth = current_val - previous_val
        retention = min(current_val, previous_val)

        return weight * (
            self.config.metric_variants['Adoption']   * adoption +
            self.config.metric_variants['Growth']     * growth +
            self.config.metric_variants['Retention']  * retention
        )


def load_configs(config_path: str) -> Tuple[SimulationConfig, AllocationConfig]:
    """Load and parse configuration files."""
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    return (
        SimulationConfig.from_yaml(yaml_config),
        AllocationConfig(
            budget=yaml_config['allocation']['budget'],
            min_amount_per_project=yaml_config['allocation']['min_amount_per_project'],
            max_share_per_project=yaml_config['allocation']['max_share_per_project'],
            max_iterations=yaml_config['allocation'].get('max_iterations', 50)
        )
    )


def main():
    # Load configurations
    sim_config, alloc_config = load_configs(DEFAULT_CONFIG)
    
    # Initialize components
    data_loader = DataLoader(sim_config)
    score_calculator = ScoreCalculator(sim_config)
    
    # Process data
    raw_data = data_loader.load_data()
    filtered_data = data_loader.filter_data(raw_data)
    
    if filtered_data.empty:
        print("\nERROR: No data after filtering!")
        print(f"Looking for periods: {sim_config.periods.keys()}")
        print(f"Looking for metrics: {sim_config.metrics.keys()}")
        return
    
    # Calculate scores and allocate
    normalized_scores = score_calculator.compute_project_scores(filtered_data)
    final_allocations = allocate_with_constraints(normalized_scores, alloc_config)


if __name__ == "__main__":
    main() 