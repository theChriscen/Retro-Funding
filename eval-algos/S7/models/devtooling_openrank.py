"""
devtooling_openrank.py

Model pipeline:

1) Instantiate dataclasses with defaults
2) Load YAML config
3) Load raw data
4) Pre-process (generate graphs)
5) Run OpenRank
6) Add metrics and metadata
7) Normalize scores
8) Serialize final results
9) Return 'analysis'
"""

from dataclasses import dataclass
import numpy as np
from openrank_sdk import EigenTrust
import pandas as pd
from typing import Dict, Any, Tuple
import yaml


@dataclass
class DataSnapshot:
    data_dir: str
    onchain_projects_file: str
    devtooling_projects_file: str
    package_links_file: str
    developers_to_repositories_file: str

@dataclass
class SimulationConfig:
    alpha_developer_to_repository: float
    alpha_package_dependency: float
    onchain_project_pretrust_weights: Dict[str, float]
    dependency_source_weights: Dict[str, float]
    binary_event_weights: Dict[str, float]
    total_event_weights: Dict[str, float]
    link_type_weights: Dict[str, float]
    eligibility_thresholds: Dict[str, int]

class DevtoolingCalculator:
    """
    Encapsulates logic for generating OpenRank graphs.
    Produces an 'analysis' dict of all intermediate DataFrames.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.analysis = {}

    def run_analysis(
        self,
        df_onchain_projects: pd.DataFrame,
        df_devtooling_projects: pd.DataFrame,
        df_package_links: pd.DataFrame,
        df_developers_to_repositories: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Main pipeline producing an 'analysis' dictionary with all intermediate steps.
        """
        self.analysis = {
            'onchain_projects': df_onchain_projects,
            'devtooling_projects': df_devtooling_projects,
            'package_links': df_package_links,
            'developers_to_repositories': df_developers_to_repositories
        }

        # Step 1: Generate pretrust scores (for onchain projects)
        self._generate_onchain_project_pretrust_scores()

        # Step 2: Weight package links graph (for local trust scores)
        self._weight_package_links()

        # Step 3: Weight developer to repository links (for local trust scores)
        self._weight_developer_to_repository_links()
        
        # Step 4: Score graphs and combine into a single matrix
        self._run_openrank()

        # Step 5: Add metrics and metadata
        self._add_metrics_and_metadata()

        # Step 6: Normalize into final scores
        self._normalize_scores()

        # Step 7: Serialize final results
        self._serialize_results()

        # Step 8: Return final results
        return self.analysis

    # --------------------------------------------------------------------
    # Internal methods
    # --------------------------------------------------------------------

    def _minmax_scale(self, values: np.ndarray) -> np.ndarray:
        """Min-max scaling."""
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax == vmin:
            return np.full_like(values, 0.5)
        return (values - vmin) / (vmax - vmin)

    def _generate_onchain_project_pretrust_scores(self) -> None:
        """Generate pretrust scores for onchain projects."""

        df_onchain_projects = self.analysis['onchain_projects'].copy()
        onchain_project_pretrust_weights = self.config.onchain_project_pretrust_weights

        pretrust_cols = list(onchain_project_pretrust_weights.keys())
        df_onchain_projects[pretrust_cols] = df_onchain_projects[pretrust_cols].apply(self._minmax_scale)
        df_onchain_projects['v'] = df_onchain_projects[pretrust_cols].sum(axis=1)
        df_onchain_projects['v'] /= df_onchain_projects['v'].sum()
        df_onchain_projects.rename(columns={'project_id': 'i'}, inplace=True)

        self.analysis['onchain_projects_pretrust_scores'] = df_onchain_projects[['i', 'v']]
    
    def _weight_package_links(self, default_weight: float = 1.0) -> None:
        """Weight package links into a graph."""
        
        df_package_links = self.analysis['package_links'].copy()
        dependency_source_weights = self.config.dependency_source_weights
        
        # Create a new column 'edge_weight' that assigns a weight based on dependency_source.
        df_package_links['edge_weight'] = df_package_links['dependency_source'].apply(
            lambda ds: dependency_source_weights.get(ds, default_weight)
        )

        # Aggregate the links so that each onchain → devtooling pair gets a weight.
        self.analysis['project_package_links_weighted'] = (
            df_package_links
            .groupby(['onchain_builder_project_id', 'devtooling_project_id'], as_index=False)
            .agg({'edge_weight': 'max'}) # TODO: Consider something more sophisticated that account for multiple valuable repos.
            .rename(columns={
                'onchain_builder_project_id': 'i',
                'devtooling_project_id': 'j',
                'edge_weight': 'v'
            })
            .assign(link_type='package_dependency')
        )


    def _weight_developer_to_repository_links(self, default_weight: float = 1.0) -> None:
        """Weight developer to repository links into a graph."""

        # Get relevant properties
        onchain_project_ids = self.analysis['onchain_projects']['project_id'].unique()
        devtooling_project_ids = self.analysis['devtooling_projects']['project_id'].unique()
        df_devs_to_repositories = self.analysis['developers_to_repositories']
        binary_event_weights = self.config.binary_event_weights
        total_event_weights = self.config.total_event_weights

        # Convert last_event to datetime if it's not already
        df_devs_to_repositories['last_event'] = pd.to_datetime(df_devs_to_repositories['last_event'])

        # Define a decay function
        ref_time = df_devs_to_repositories['last_event'].max()
        age_decay = lambda x: 0.5 ** ((ref_time - x).total_seconds() / 31536000)

        # Part 1: Give trust from onchain projects to their developers
        df_devs_to_onchain_projects = (
            df_devs_to_repositories
            .query("project_id in @onchain_project_ids and event_type == 'COMMIT_CODE'")
            .groupby(['developer_id', 'project_id'], as_index=False)
            .agg({
                'last_event': 'max',
                'total_events': lambda x: 1 + np.log(x.sum() + 1)
            })
        )
        df_devs_to_onchain_projects['age_factor'] = df_devs_to_onchain_projects['last_event'].apply(age_decay)
        df_devs_to_onchain_projects['v'] = df_devs_to_onchain_projects['age_factor'] * df_devs_to_onchain_projects['total_events']

        # Part 2: Give trust from devtooling projects to their developers
        df_devs_to_devtooling_projects = (
            df_devs_to_repositories
            .query("project_id in @devtooling_project_ids")
            .groupby(['developer_id', 'project_id', 'event_type'], as_index=False)
            .agg({
                'last_event': 'max', 
                'total_events': lambda x: 1 + np.log(x.sum() + 1)
            })
        )
        # Define a helper function for weighting events
        def weight_event(event_type, amount, age_factor):
            if event_type in binary_event_weights:
                return binary_event_weights[event_type] * age_factor
            elif event_type in total_event_weights:
                return amount * total_event_weights[event_type] * age_factor
            else:
                return 0
        
        df_devs_to_devtooling_projects['age_factor'] = df_devs_to_devtooling_projects['last_event'].apply(age_decay)
        df_devs_to_devtooling_projects['v'] = df_devs_to_devtooling_projects.apply(
            lambda row: weight_event(row['event_type'], row['total_events'], row['age_factor']),
            axis=1
        )

        # Part 3: Combine the two dataframes into a bipartite graph
        df_combined = pd.concat([
            df_devs_to_onchain_projects.rename(columns={'project_id': 'i', 'developer_id': 'j'})[['i', 'j', 'v']], 
            df_devs_to_devtooling_projects.rename(columns={'developer_id': 'i', 'project_id': 'j'})[['i', 'j', 'v']]
        ], axis=0, ignore_index=True, sort=False)
        df_combined = df_combined[df_combined['v'] > 0]
        df_combined['link_type'] = 'developer_to_repository'
        self.analysis['developer_to_repository_links_weighted'] = df_combined
        

    def _run_openrank(self) -> None:
        """Run OpenRank on the weighted graphs."""
        
        et = EigenTrust()

        pretrust_scores = self.analysis['onchain_projects_pretrust_scores'].to_dict(orient='records')
        localtrust_scores_package_links = self.analysis['project_package_links_weighted'].to_dict(orient='records')
        localtrust_scores_developer_links = self.analysis['developer_to_repository_links_weighted'].to_dict(orient='records')

        s1 = et.run_eigentrust(localtrust_scores_package_links, pretrust_scores, alpha=self.config.alpha_package_dependency)
        s2 = et.run_eigentrust(localtrust_scores_developer_links, pretrust_scores, alpha=self.config.alpha_developer_to_repository)

        self.analysis['project_openrank_scores'] = pd.concat([
            pd.DataFrame(s1, columns=['i', 'v']).set_index('i').rename(columns={'v': 'v_package_links'}),
            pd.DataFrame(s2, columns=['i', 'v']).set_index('i').rename(columns={'v': 'v_developer_links'})
        ], axis=1)


    def _add_metrics_and_metadata(self) -> None:
        """Add metrics and metadata."""

        df_results = self.analysis['devtooling_projects'].copy()
        eligibility_thresholds = self.config.eligibility_thresholds 

        # Count number of onchain projects with package links
        package_links = self.analysis['project_package_links_weighted']
        df_results['num_projects_with_package_links'] = df_results['project_id'].apply(
            lambda pid: len(package_links[package_links['j'] == pid]['i'].unique())
        )

        # Count number of onchain projects with developer links 
        dev_links = self.analysis['developer_to_repository_links_weighted']
        df_results['num_projects_with_dev_links'] = df_results['project_id'].apply(
            lambda pid: len(dev_links[dev_links['j'] == pid]['i'].unique())
        )

        # Count number of onchain developers with links to the devtooling project
        df_results['num_onchain_developers_with_links'] = df_results['project_id'].apply(
            lambda pid: len(self.analysis['developers_to_repositories'][
                self.analysis['developers_to_repositories']['project_id'] == pid
            ]['developer_id'].unique())
        )

        # Determine eligibility based on thresholds
        df_results['is_eligible'] = (
            (df_results['num_projects_with_package_links'] >= eligibility_thresholds['num_projects_with_package_links']) |
            (
                (df_results['num_projects_with_dev_links'] >= eligibility_thresholds['num_projects_with_dev_links']) &
                (df_results['num_onchain_developers_with_links'] >= eligibility_thresholds['num_onchain_developers_with_links'])
            )
        )
        df_results['is_eligible'] = df_results['is_eligible'].astype(int)

        self.analysis['devtooling_project_results'] = df_results


    def _normalize_scores(self) -> None:
        """Normalize scores."""

        # Merge OpenRank scores
        df_results = self.analysis['devtooling_project_results'].copy()
        df_scores = self.analysis['project_openrank_scores'].fillna(0).copy()
        df_results = df_results.merge(df_scores, left_on='project_id', right_on='i')
        
        # Apply link type weights to each OpenRank score
        link_type_weights = self.config.link_type_weights
        df_scores['v_package_links'] = df_scores['v_package_links'] * link_type_weights['package_dependency']
        df_scores['v_developer_links'] = df_scores['v_developer_links'] * link_type_weights['developer_to_repository']
        aggregated_scores = df_scores.sum(axis=1)
        aggregated_scores.name = 'v_aggregated'
        df_results = df_results.join(aggregated_scores, on='project_id')

        # Apply eligibility mask
        df_results['v_aggregated'] = df_results['v_aggregated'] * df_results['is_eligible']

        # Normalize scores
        df_results['v_aggregated'] /= df_results['v_aggregated'].sum()
        
        # Store results
        self.analysis['devtooling_project_results'] = df_results


    def _serialize_results(self) -> None:
        """Serialize results."""
        self.analysis['devtooling_project_results'] = self.analysis['devtooling_project_results'].sort_values(by='v_aggregated', ascending=False)


# ------------------------------------------------------------------------
# Load config & data
# ------------------------------------------------------------------------

def load_config(config_path: str) -> Tuple[DataSnapshot, SimulationConfig]:
    """
    Load configuration from YAML.
    Returns (DataSnapshot, SimulationConfig).
    """
    with open(config_path, 'r') as f:
        ycfg = yaml.safe_load(f)

    ds = DataSnapshot(
        data_dir=ycfg['data_snapshot'].get('data_dir', "eval-algos/S7/data/devtooling_testing2"),
        onchain_projects_file=ycfg['data_snapshot'].get('onchain_projects_file', "onchain_projects.csv"),
        devtooling_projects_file=ycfg['data_snapshot'].get('devtooling_projects_file', "devtooling_projects.csv"),
        package_links_file=ycfg['data_snapshot'].get('package_links_file', "package_links.csv"),
        developers_to_repositories_file=ycfg['data_snapshot'].get('developers_to_repositories_file', "developers_to_repositories.csv")
    )

    # Load simulation config directly from YAML
    sim = ycfg.get('simulation', {})
    sc = SimulationConfig(
        alpha_developer_to_repository=sim.get('alpha_developer_to_repository', 0.5),
        alpha_package_dependency=sim.get('alpha_package_dependency', 0.5),
        onchain_project_pretrust_weights=sim.get('onchain_project_pretrust_weights', {}),
        dependency_source_weights=sim.get('dependency_source_weights', {}),
        binary_event_weights=sim.get('binary_event_weights', {}),
        total_event_weights=sim.get('total_event_weights', {}),
        link_type_weights=sim.get('link_type_weights', {}),
        eligibility_thresholds=sim.get('eligibility_thresholds', {})
    )

    return ds, sc

def load_data(ds: DataSnapshot) -> pd.DataFrame:
    """
    Load raw CSV data, merge into single DataFrame.
    """
    def path(x: str):
        return f"{ds.data_dir}/{x}"

    df_onchain_projects = pd.read_csv(path(ds.onchain_projects_file))
    df_devtooling_projects = pd.read_csv(path(ds.devtooling_projects_file))
    df_package_links = pd.read_csv(path(ds.package_links_file))
    df_developers_to_repositories = pd.read_csv(path(ds.developers_to_repositories_file))

    return df_onchain_projects, df_devtooling_projects, df_package_links, df_developers_to_repositories


# ------------------------------------------------------------------------
# Main pipeline entry-point & packaging
# ------------------------------------------------------------------------

def run_simulation(config_path: str) -> Dict[str, Any]:
    """
    Main pipeline entry-point.
    """
    # Load config & data
    ds, sim_cfg = load_config(config_path)
    data = load_data(ds)

    # Create calculator and run analysis
    calculator = DevtoolingCalculator(sim_cfg)
    analysis = calculator.run_analysis(*data)

    # Store config references
    analysis["data_snapshot"] = ds
    analysis["simulation_config"] = sim_cfg

    return analysis


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

    out_path = f"{ds.data_dir}/devtooling_openrank_results.csv"
    analysis["devtooling_project_results"].to_csv(out_path, index=False)
    print(f"Saved devtooling openrank results to {out_path}")


# ------------------------------------------------------------------------
# main()
# ------------------------------------------------------------------------

def main():
    """
    Standard entry-point for running this script from CLI.
    """
    config_path = 'eval-algos/S7/weights/devtooling_openrank_testing.yaml'
    analysis = run_simulation(config_path)
    save_results(analysis)


if __name__ == "__main__":
    main()