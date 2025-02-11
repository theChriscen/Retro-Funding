"""
onchain_builders.py

Standardized 8-step model pipeline:

1) Instantiate dataclasses with defaults
2) Load YAML config
3) Load raw data
4) Pre-process (pivot, etc.)
5) Run score calculator
6) Package into 'analysis'
7) Serialize final results
8) Return 'analysis'
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import yaml


@dataclass
class DataSnapshot:
    data_dir: str = 'eval-algos/S7/data/onchain_testing'
    projects_file: str = 'projects_v1.csv'
    metrics_file: str = 'onchain_metrics_by_project.csv'

@dataclass
class SimulationConfig:
    periods: Dict[str, str] = field(default_factory=lambda: {
        'Dec 2024': 'previous',
        'Jan 2025': 'current'
    })
    metrics: Dict[str, float] = field(default_factory=lambda: {
        'transaction_count_bot_filtered': 0.30,
        'monthly_active_farcaster_users': 0.10,
        'transaction_gas_fee': 0.30,
        'trace_count': 0.30
    })
    chains: Dict[str, float] = field(default_factory=lambda: {
        'BASE': 1.0,
        'OPTIMISM': 1.0
    })
    metric_variants: Dict[str, float] = field(default_factory=lambda: {
        'Adoption': 0.20,
        'Growth': 0.10,
        'Retention': 0.70
    })
    normalization: Dict[str, str] = field(default_factory=lambda: {
        'method': 'minmax',
        'robust_scale_limit': 3,
        'center_value': 0.5
    })


class OnchainBuildersCalculator:
    """
    Encapsulates logic for pivoting and computing metric-based scores.
    Produces an 'analysis' dict of all intermediate DataFrames.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def run_analysis(self, df_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Main pipeline producing an 'analysis' dictionary with all intermediate steps.
        """
        analysis = {}

        # Step 1: Pivot raw data
        analysis["pivoted_raw_metrics_by_chain"] = self._filter_and_pivot_raw_metrics_by_chain(df_data)

        # Step 2: Sum & weight
        analysis["pivoted_raw_metrics_weighted_by_chain"] = self._sum_and_weight_raw_metrics_by_chain(
            analysis["pivoted_raw_metrics_by_chain"]
        )

        # Step 3: Calculate metric variants
        analysis["pivoted_metric_variants"] = self._calculate_metric_variants(
            analysis["pivoted_raw_metrics_weighted_by_chain"]
        )

        # Step 4: Normalize
        analysis["normalized_metric_variants"] = self._normalize_metric_variants(
            analysis["pivoted_metric_variants"]
        )

        # Step 5: Apply weights
        analysis["weighted_metric_variants"] = self._apply_weights_to_metric_variants(
            analysis["normalized_metric_variants"]
        )

        # Step 6: Aggregate final scores
        analysis["aggregated_project_scores"] = self._aggregate_metric_variants(
            analysis["weighted_metric_variants"]
        )

        return analysis

    # --------------------------------------------------------------------
    # Internal methods
    # --------------------------------------------------------------------

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
        """Sum and weight raw metrics by chain weighting."""
        chain_weights = pd.Series(self.config.chains)
        weighted_df = (
            df.mul(df.index.get_level_values('chain').map(chain_weights).fillna(1.0), axis=0)
              .groupby(['project_id', 'project_name', 'display_name'])
              .sum()
        )
        return weighted_df

    def _calculate_metric_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Adoption, Growth, Retention from current vs. previous period."""
        current_period = next(k for k, v in self.config.periods.items() if v == 'current')
        previous_period = next(k for k, v in self.config.periods.items() if v == 'previous')

        variant_scores = {}
        for metric in self.config.metrics.keys():
            current_vals = df[(current_period, metric)].fillna(0)
            prev_vals = df[(previous_period, metric)].fillna(0)

            variant_scores[(metric, 'Adoption')] = current_vals
            variant_scores[(metric, 'Growth')] = current_vals - prev_vals
            variant_scores[(metric, 'Retention')] = pd.concat([current_vals, prev_vals], axis=1).min(axis=1)

        return pd.DataFrame(variant_scores)

    def _normalize_metric_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to each metric variant column."""
        method = self.config.normalization.get('method', 'minmax')
        df_norm = df.copy()
        for col in df_norm.columns:
            if method == 'minmax':
                df_norm[col] = self._minmax_scale(df_norm[col].values)
            elif method == 'robust':
                df_norm[col] = self._robust_scale(df_norm[col].values)
        return df_norm

    def _minmax_scale(self, values: np.ndarray) -> np.ndarray:
        """Min-max scaling with fallback to center_value if no range."""
        C = self.config.normalization.get('center_value', 0.5)
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax == vmin:
            return np.full_like(values, C)
        return (values - vmin) / (vmax - vmin)

    def _robust_scale(self, values: np.ndarray) -> np.ndarray:
        """Robust scaling using IQR, with optional clipping."""
        C = self.config.normalization.get('center_value', 0.5)
        lim = self.config.normalization.get('robust_scale_limit', 3)
        median = np.nanmedian(values)
        q1, q3 = np.nanpercentile(values, 25), np.nanpercentile(values, 75)
        iqr = q3 - q1
        if iqr == 0:
            return np.full_like(values, C)

        scaled = (values - median) / iqr
        clipped = np.clip(scaled, -lim, lim)
        return (clipped + lim) / (2 * lim)

    def _apply_weights_to_metric_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multiply metric & variant weights onto normalized data."""
        out = df.copy()
        for metric, m_weight in self.config.metrics.items():
            for variant, v_weight in self.config.metric_variants.items():
                out[(metric, variant)] *= (m_weight * v_weight)
        return out

    def _aggregate_metric_variants(self, df: pd.DataFrame, method: str = 'geometric_mean') -> pd.DataFrame:
        """Combine variant columns into a single project score (sum or geometric_mean)."""
        out = df.copy()
        if method == 'sum':
            out['project_score'] = df.sum(axis=1)
        elif method == 'geometric_mean':
            epsilon = 1e-9
            out['project_score'] = np.exp(np.log(df + epsilon).mean(axis=1))
        else:
            raise ValueError(f"Invalid aggregation method: {method}")
        return out


# ------------------------------------------------------------------------
# Load config & data
# ------------------------------------------------------------------------

def load_config(config_path: str) -> Tuple[DataSnapshot, SimulationConfig]:
    """
    Load configuration from YAML (or default if missing).
    Returns (DataSnapshot, SimulationConfig).
    """
    try:
        with open(config_path, 'r') as f:
            ycfg = yaml.safe_load(f)
    except FileNotFoundError:
        return DataSnapshot(), SimulationConfig()

    ds = DataSnapshot(
        projects_file=ycfg['data_snapshot'].get('projects_file', "projects.csv"),
        metrics_file=ycfg['data_snapshot'].get('metrics_file', "metrics.csv")
    )

    sc = SimulationConfig(
        periods=ycfg['simulation']['periods'],
        metrics=ycfg['simulation']['metrics'],
        chains=ycfg['simulation']['chains'],
        metric_variants=ycfg['simulation']['metric_variants'],
        normalization=ycfg['simulation']['normalization']
    )

    return ds, sc


def load_data(ds: DataSnapshot) -> pd.DataFrame:
    """
    Load raw CSV data, merge into single DataFrame.
    """
    def path(x: str):
        return f"{ds.data_dir}/{x}"

    df_projects = pd.read_csv(path(ds.projects_file))
    df_metrics = pd.read_csv(path(ds.metrics_file))

    df_metrics['measurement_period'] = pd.to_datetime(df_metrics['sample_date']).dt.strftime('%b %Y')
    return df_metrics.merge(df_projects, on='project_id', how='left')


# ------------------------------------------------------------------------
# Main pipeline entry-point & packaging
# ------------------------------------------------------------------------

def run_simulation(config_path: str) -> Dict[str, Any]:
    """
    Orchestrates the entire pipeline:

      1) Instantiate defaults
      2) load_config
      3) load_data
      4) Pre-process & pivot
      5) Run score calculations
      6) Assemble 'analysis'
      7) (Optionally) Save
      8) Return 'analysis'
    """
    ds, sim_cfg = load_config(config_path)
    df_data = load_data(ds)

    # Create calculator and run analysis
    calculator = OnchainBuildersCalculator(sim_cfg)
    analysis = calculator.run_analysis(df_data)

    # Derive final weighting
    scores_df = analysis['aggregated_project_scores']
    scores_series = scores_df['project_score']
    normalized_series = scores_series / scores_series.sum()
    normalized_series.name = 'Weighted Score'

    # Flatten pivot
    df_pivoted_weighted = _flatten_columns(analysis['pivoted_raw_metrics_weighted_by_chain'])
    df_variants = _flatten_columns(analysis['pivoted_metric_variants'])

    final_df = (
        df_pivoted_weighted
        .join(df_variants)
        .join(normalized_series)
        .sort_values('Weighted Score', ascending=False)
    )
    analysis["final_results"] = final_df

    # Store config references
    analysis["data_snapshot"] = ds
    analysis["simulation_config"] = sim_cfg

    return analysis


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens multi-level columns after pivot/weighting."""
    def _flat(col):
        if isinstance(col, tuple):
            return ": ".join(str(c) for c in col)
        return str(col)
    out = df.copy()
    out.columns = [_flat(c) for c in df.columns]
    return out


# ------------------------------------------------------------------------
# Serialize
# ------------------------------------------------------------------------

def save_results(analysis: Dict[str, Any]) -> None:
    """
    Write final results to a CSV if data_snapshot is available.
    """
    ds = analysis.get("data_snapshot")
    if ds is None:
        print("No DataSnapshot found; skipping file output.")
        return

    out_path = f"{ds.data_dir}/onchain_builders_testing_results.csv"
    analysis["final_results"].to_csv(out_path, index=True)
    print(f"Saved onchain builders results to {out_path}")


# ------------------------------------------------------------------------
# main()
# ------------------------------------------------------------------------

def main():
    """
    Standard entry-point for running this script from CLI.
    """
    config_path = 'eval-algos/S7/weights/onchain_builders_testing.yaml'
    analysis = run_simulation(config_path)
    save_results(analysis)


if __name__ == "__main__":
    main()