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

    # --------------------------------------------------------------------
    # Main pipeline
    # --------------------------------------------------------------------

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
        # Store raw frames in analysis
        self.analysis = {
            'onchain_projects': df_onchain_projects,
            'devtooling_projects': df_devtooling_projects,
            'package_links': df_package_links,
            'developers_to_repositories': df_developers_to_repositories
        }

        # 1: Generate onchain pretrust
        self._generate_onchain_project_pretrust_scores()

        # 2: Build a single, unified edge list (onchain->devtooling, onchain->developer, dev->devtooling)
        self._build_unified_edge_list()

        # 3: Run a single EigenTrust pass on that unified adjacency
        self._run_openrank()

        # 4: Add metrics & metadata
        self._add_metrics_and_metadata()

        # 5: Normalize & finalize devtooling scores
        self._normalize_scores()

        # 6: Serialize final results
        self._serialize_results()

        # 7: Return final analysis
        return self.analysis

    # --------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------

    @staticmethod
    def _minmax_scale(values: pd.Series) -> pd.Series:
        """Min-max scaling of a pandas Series."""
        vmin, vmax = values.min(), values.max()
        if vmax == vmin:
            # If everything is identical, just return 0.5
            return pd.Series([0.5]*len(values), index=values.index)
        return (values - vmin) / (vmax - vmin)

    def _generate_onchain_project_pretrust_scores(self) -> None:
        """
        Generate pretrust vector for onchain projects.
        We min-max scale each column and then sum them up,
        then normalize so they sum to 1 across all onchain projects.
        """
        df_onchain = self.analysis['onchain_projects'].copy()
        wts = self.config.onchain_project_pretrust_weights

        # Only operate on the columns we have weights for
        pretrust_cols = list(wts.keys())
        # Min-max scale each column
        for col in pretrust_cols:
            if col in df_onchain.columns:
                df_onchain[col] = self._minmax_scale(df_onchain[col])
            else:
                # If the column doesn't exist, fill with 0
                df_onchain[col] = 0.0

        # Weighted sum across all pretrust columns
        df_onchain['v_raw'] = 0.0
        for c in pretrust_cols:
            df_onchain['v_raw'] += df_onchain[c] * wts[c]

        # Normalize so total across all onchain = 1
        total = df_onchain['v_raw'].sum()
        if total > 0:
            df_onchain['v_raw'] /= total
        else:
            # Edge case: if all zero, just distribute uniformly
            df_onchain['v_raw'] = 1.0 / len(df_onchain)

        # This final 'v' is what we feed as pretrust
        df_onchain = df_onchain.rename(columns={'project_id': 'i'})
        df_onchain = df_onchain[['i', 'v_raw']].copy()
        df_onchain.rename(columns={'v_raw': 'v'}, inplace=True)

        self.analysis['onchain_projects_pretrust_scores'] = df_onchain

    def _build_unified_edge_list(self) -> None:
        """
        Build a single DataFrame of edges with columns:
        [i, j, link_type, v_raw, v_final]
        Tracks 'v_raw' (before scaling) and 'v_final' (after minmax).
        
        Edges:
          - onchain -> devtooling (package dependency)
          - onchain -> developer   (optional, if you want to reward devs who built onchain)
          - developer -> devtooling (the single most valuable event + time decay)
        """
        ###################################################
        # PART A: onchain->devtooling via package usage
        ###################################################
        df_pkglinks = self.analysis['package_links'].copy()
        ds_wts = self.config.dependency_source_weights

        # 1) Map each row's dependency_source to a weight (fallback = 1.0 if not found)
        df_pkglinks['source_wt'] = df_pkglinks['dependency_source'].apply(
            lambda ds: ds_wts.get(ds, 1.0)
        )

        # 2) Group by (onchain_builder_project_id, devtooling_project_id, dependency_artifact_id)
        # to get sum of the source_wt across that single "repo"
        # Then sum these repo-level totals across all repos for that pair.
        df_repo_sums = (
            df_pkglinks
            .groupby(['onchain_builder_project_id','devtooling_project_id','dependency_artifact_id'], as_index=False)
            .agg({'source_wt':'sum'})  # sum of source weights in that single repo
            .rename(columns={'source_wt':'repo_weight'})
        )

        df_pair_sums = (
            df_repo_sums
            .groupby(['onchain_builder_project_id','devtooling_project_id'], as_index=False)
            .agg({'repo_weight':'sum'})
            .rename(columns={'repo_weight':'v_raw'})
        )

        df_pair_sums['link_type'] = 'package_dependency'
        df_pair_sums = df_pair_sums.rename(columns={
            'onchain_builder_project_id': 'i',
            'devtooling_project_id': 'j'
        })

        ###################################################
        # PART B: onchain->developer (optional)
        ###################################################
        df_devrepo = self.analysis['developers_to_repositories'].copy()

        # Identify onchain project IDs
        onchain_ids = set(self.analysis['onchain_projects']['project_id'].unique())

        df_onchain_dev = df_devrepo[df_devrepo['project_id'].isin(onchain_ids)].copy()
        df_onchain_dev['last_event'] = pd.to_datetime(df_onchain_dev['last_event'])

        # Time decay reference
        ref_time = df_onchain_dev['last_event'].max()
        def time_decay(t):
            if pd.isnull(t):
                return 1.0
            return 0.5 ** ((ref_time - t).total_seconds() / 31536000)

        # Only consider commits for onchain->developer
        df_onchain_dev = df_onchain_dev[df_onchain_dev['event_type'] == 'COMMIT_CODE'].copy()
        df_onchain_dev['base_wt'] = 1 + np.log(df_onchain_dev['total_events'] + 1)
        df_onchain_dev['decay'] = df_onchain_dev['last_event'].apply(time_decay)
        df_onchain_dev['v_calc'] = df_onchain_dev['base_wt'] * df_onchain_dev['decay']

        df_onchain_devmax = (
            df_onchain_dev
            .groupby(['project_id','developer_id'], as_index=False)['v_calc'].max()
            .rename(columns={'project_id':'i','developer_id':'j','v_calc':'v_raw'})
        )
        df_onchain_devmax['link_type'] = 'onchain_to_developer'

        ###################################################
        # PART C: developer->devtooling (most valuable event + time decay)
        ###################################################
        devtool_ids = set(self.analysis['devtooling_projects']['project_id'].unique())
        df_dev2tool = df_devrepo[df_devrepo['project_id'].isin(devtool_ids)].copy()

        # Convert last_event to datetime
        df_dev2tool['last_event'] = pd.to_datetime(df_dev2tool['last_event'])

        def time_decay2(t):
            if pd.isnull(t) or pd.isnull(ref_time):
                return 1.0
            return 0.5 ** ((ref_time - t).total_seconds() / 31536000)

        def event_weight(evt_type, total_evt):
            # If it's in binary_event_weights, use that
            if evt_type in self.config.binary_event_weights:
                return self.config.binary_event_weights[evt_type]
            # If it's in total_event_weights, do log scale + config weight
            elif evt_type in self.config.total_event_weights:
                return (1 + np.log(total_evt + 1)) * self.config.total_event_weights[evt_type]
            else:
                return 0.0

        df_dev2tool['time_decay'] = df_dev2tool['last_event'].apply(time_decay2)
        df_dev2tool['base_wt'] = df_dev2tool.apply(
            lambda row: event_weight(row['event_type'], row['total_events']),
            axis=1
        )
        df_dev2tool['v_calc'] = df_dev2tool['base_wt'] * df_dev2tool['time_decay']

        df_dev2toolmax = (
            df_dev2tool
            .groupby(['developer_id','project_id'], as_index=False)['v_calc'].max()
            .rename(columns={'developer_id':'i','project_id':'j','v_calc':'v_raw'})
        )
        df_dev2toolmax['link_type'] = 'developer_to_devtool'

        ###################################################
        # Combine all edges
        ###################################################
        df_edges = pd.concat([
            df_pair_sums[['i','j','link_type','v_raw']],
            df_onchain_devmax[['i','j','link_type','v_raw']],
            df_dev2toolmax[['i','j','link_type','v_raw']],
        ], ignore_index=True)

        self.analysis['unified_edges'] = df_edges.copy()

        ###################################################
        # Scale v_raw -> v_final
        ###################################################
        df_edges['v_final'] = self._minmax_scale(df_edges['v_raw'])

        # Then multiply by link type weights if desired
        link_type_wts = self.config.link_type_weights
        df_edges['v_final'] = df_edges.apply(
            lambda row: row['v_final'] * link_type_wts.get(row['link_type'], 1.0),
            axis=1
        )

        # Store final edges used in EigenTrust
        self.analysis['unified_edges_final'] = df_edges.copy()

    def _run_openrank(self) -> None:
        """Run a single EigenTrust pass on the unified edges."""
        et = EigenTrust()

        # Pretrust from onchain projects
        pretrust_scores = self.analysis['onchain_projects_pretrust_scores'].to_dict(orient='records')

        # The final edges we want to feed to EigenTrust
        df_edges_final = self.analysis['unified_edges_final'].copy()

        # Typically pick a single alpha. 
        # We'll use alpha_developer_to_repository for demonstration
        alpha = self.config.alpha_developer_to_repository

        # Convert to the format EigenTrust expects: a list of dicts with i, j, v
        edge_records = df_edges_final.apply(
            lambda row: {'i': row['i'], 'j': row['j'], 'v': row['v_final']},
            axis=1
        ).tolist()

        scores = et.run_eigentrust(edge_records, pretrust_scores, alpha=alpha)

        # Store results as a DataFrame in analysis
        df_scores = pd.DataFrame(scores, columns=['i','v']).set_index('i')
        self.analysis['project_openrank_scores'] = df_scores

    def _add_metrics_and_metadata(self) -> None:
        """
        Add existing analytics/metrics for devtooling projects, e.g. 
        num_projects_with_package_links, etc.
        """
        df_results = self.analysis['devtooling_projects'].copy()
        df_edges = self.analysis['unified_edges']  # raw edges with link_type

        # For "num_projects_with_package_links"
        df_pkg = df_edges[df_edges['link_type'] == 'package_dependency']
        df_results['num_projects_with_package_links'] = df_results['project_id'].apply(
            lambda pid: df_pkg[df_pkg['j'] == pid]['i'].nunique()
        )

        # For "num_projects_with_dev_links"
        df_dev = df_edges[df_edges['link_type'] == 'developer_to_devtool']
        df_results['num_projects_with_dev_links'] = df_results['project_id'].apply(
            lambda pid: df_dev[df_dev['j'] == pid]['i'].nunique()
        )

        # "num_onchain_developers_with_links"
        df_devrepo = self.analysis['developers_to_repositories']
        onchain_project_ids = set(self.analysis['onchain_projects']['project_id'].unique())
        devs_onchain = set(df_devrepo[df_devrepo['project_id'].isin(onchain_project_ids)]['developer_id'].unique())

        df_results['num_onchain_developers_with_links'] = df_results['project_id'].apply(
            lambda pid: len(
                set(df_dev[df_dev['j'] == pid]['i'].unique()) & devs_onchain
            )
        )

        # Apply eligibility thresholds
        elig = self.config.eligibility_thresholds
        df_results['is_eligible'] = (
            (df_results['num_projects_with_package_links'] >= elig['num_projects_with_package_links']) |
            (
                (df_results['num_projects_with_dev_links'] >= elig['num_projects_with_dev_links']) &
                (df_results['num_onchain_developers_with_links'] >= elig['num_onchain_developers_with_links'])
            )
        ).astype(int)

        self.analysis['devtooling_project_results'] = df_results

    def _normalize_scores(self) -> None:
        """
        Merge the eigen trust results with devtooling projects,
        apply eligibility mask, then do a final normalization so that
        v_aggregated sums to 1 across devtooling projects.
        """
        df_results = self.analysis['devtooling_project_results'].copy()
        df_scores = self.analysis['project_openrank_scores'].copy()  # i -> v

        # Merge
        df_results = df_results.merge(df_scores, left_on='project_id', right_on='i', how='left')
        df_results['v'].fillna(0.0, inplace=True)

        # Apply eligibility
        df_results['v_aggregated'] = df_results['v'] * df_results['is_eligible']

        # Final normalization across devtooling projects
        total = df_results['v_aggregated'].sum()
        if total > 0:
            df_results['v_aggregated'] /= total

        # Store
        self.analysis['devtooling_project_results'] = df_results

    def _serialize_results(self) -> None:
        """
        Sort devtooling projects by final 'v_aggregated' and store in analysis.
        """
        df = self.analysis['devtooling_project_results'].sort_values(by='v_aggregated', ascending=False)
        self.analysis['devtooling_project_results'] = df


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

def load_data(ds: DataSnapshot) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV data, returning the four expected DataFrames.
    """
    def path(x: str):
        return f"{ds.data_dir}/{x}"

    df_onchain_projects = pd.read_csv(path(ds.onchain_projects_file))
    df_devtooling_projects = pd.read_csv(path(ds.devtooling_projects_file))
    df_package_links = pd.read_csv(path(ds.package_links_file))
    df_developers_to_repositories = pd.read_csv(path(ds.developers_to_repositories_file))

    return (
        df_onchain_projects,
        df_devtooling_projects,
        df_package_links,
        df_developers_to_repositories
    )


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