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

# Suppress the specific EigenTrust deprecation warning, which isn't supported yet
import warnings
warnings.filterwarnings('ignore', message='Defaulting to the \'raw\' score scale*')

@dataclass
class DataSnapshot:
    data_dir: str
    onchain_projects_file: str
    devtooling_projects_file: str
    package_links_file: str
    developers_to_repositories_file: str

@dataclass
class SimulationConfig:
    alpha: float
    time_decay: Dict[str, float]
    onchain_project_pretrust_weights: Dict[str, float]
    dependency_source_weights: Dict[str, float]
    git_event_weights: Dict[str, float]
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

        # 4: Add metrics & metadata useful for QA and eligibility filters
        self._add_metrics_and_metadata()

        # 5: Filter out non-eligible projects, then normalize into final scores
        self._normalize_scores()

        # 6: Serialize final results
        self._serialize_results()

        return self.analysis

    # --------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------

    @staticmethod
    def _minmax_scale(values: pd.Series) -> pd.Series:
        """Min-max scaling of a series of values to [0, 1]."""
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

        # Only operate on the columns we have weights for in our YAML config
        pretrust_cols = list(wts.keys())
        # Min-max scale each column
        for col in pretrust_cols:
            if col in df_onchain.columns:
                df_onchain[col] = self._minmax_scale(df_onchain[col])
            else:
                # If the column doesn't exist, fill with 0
                df_onchain[col] = 0.0

        # Weighted sum across all pretrust columns
        df_onchain['v'] = 0.0
        for c in pretrust_cols:
            df_onchain['v'] += df_onchain[c] * wts[c]

        # Normalize so total across all onchain = 1
        total = df_onchain['v'].sum()
        if total > 0:
            df_onchain['v'] /= total
        else:
            # Edge case: if all zero, just distribute uniformly
            df_onchain['v'] = 1.0 / len(df_onchain)

        # This final 'v' is what we feed as pretrust
        df_onchain = df_onchain.rename(columns={'project_id': 'i'})
        df_onchain = df_onchain[['i', 'v']].copy()

        self.analysis['onchain_projects_pretrust_scores'] = df_onchain

    def _build_unified_edge_list(self) -> None:
        """
        Build a single DataFrame of edges with columns:
        [i, j, link_type, v_raw, v_final]
        Tracks 'v_raw' (before scaling) and 'v_final' (after minmax).
        
        Edges:
          - onchain -> devtooling   (onchain projects confer weight to package links)
          - onchain -> developer    (onchain projects confer weight to their own devs)
          - developer -> devtooling (git events from onchain devs confer weight to devtooling repos)
        """

        # Time decay settings
        commit_decay = self.config.time_decay['commit_to_onchain_repo']
        event_decay = self.config.time_decay['event_to_devtooling_repo']

        #--------------------------------------------------------------------
        # PART A: onchain -> devtooling via package usage
        #--------------------------------------------------------------------
        df_pkglinks = self.analysis['package_links'].copy()
        ds_wts = self.config.dependency_source_weights

        # 1) Map each row's dependency_source to a weight (fallback = 1.0 if not found)
        df_pkglinks['source_wt'] = df_pkglinks['dependency_source'].apply(
            lambda ds: ds_wts.get(ds, 1.0)
        )

        # 2) Determine the sum of the source weights for a single devtooling repo
        df_repo_sums = (
            df_pkglinks
            .groupby(['onchain_builder_project_id','devtooling_project_id','dependency_artifact_id'], as_index=False)
            .agg({'source_wt':'sum'})  # sum of source weights in that single repo
            .rename(columns={'source_wt':'repo_weight'})
        )

        # 3) Sum the repo-level weights across all devtooling repos for a single onchain project
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

        #--------------------------------------------------------------------
        # PART B: onchain -> developer via commit events
        #--------------------------------------------------------------------
        
        # Get the last event time for all developers
        df_devrepo = self.analysis['developers_to_repositories'].copy()
        df_devrepo['first_event'] = pd.to_datetime(df_devrepo['first_event'])
        df_devrepo['last_event'] = pd.to_datetime(df_devrepo['last_event'])
        ref_time = df_devrepo['last_event'].max()

        # Identify onchain project IDs
        onchain_ids = set(self.analysis['onchain_projects']['project_id'].unique())
        df_onchain_dev = df_devrepo[df_devrepo['project_id'].isin(onchain_ids)].copy()

        # Only consider commit events for onchain->developer
        df_onchain_dev = df_onchain_dev[df_onchain_dev['event_type'] == 'COMMIT_CODE'].copy()

        # Group by developer and project, then get the min/max and total events
        df_onchain_dev = df_onchain_dev.groupby(['developer_id','project_id'], as_index=False).agg({
            'first_event': 'min',
            'last_event': 'max',
            'total_events': 'sum'
        })

        # Only consider developers with at least 3 months (90 days) between their first and last commit to a project
        df_onchain_dev['time_diff'] = df_onchain_dev['last_event'] - df_onchain_dev['first_event']
        df_onchain_dev = df_onchain_dev[df_onchain_dev['time_diff'] >= pd.Timedelta(days=90)]

        # Weight developers based on the number of events they've had on a log scale
        df_onchain_dev['base_wt'] = 1 + np.log(df_onchain_dev['total_events'])
        
        # Apply a time decay function
        def time_decay(t):
            if pd.isnull(t):
                return 1.0
            return commit_decay ** ((ref_time - t).total_seconds() / 31536000)
        df_onchain_dev['decay'] = df_onchain_dev['last_event'].apply(time_decay)
        df_onchain_dev['v_raw'] = df_onchain_dev['base_wt'] * df_onchain_dev['decay']

        # Get the value v_raw for each developer and project
        df_onchain_devmax = (
            df_onchain_dev
            .groupby(['project_id','developer_id'], as_index=False)['v_raw'].max()
            .rename(columns={'project_id':'i','developer_id':'j'})
        )
        df_onchain_devmax['link_type'] = 'onchain_to_developer'

        #--------------------------------------------------------------------
        # PART C: developer -> devtooling via git events
        #--------------------------------------------------------------------
        
        # Identify devtooling project IDs
        devtool_ids = set(self.analysis['devtooling_projects']['project_id'].unique())
        df_dev2tool = df_devrepo[df_devrepo['project_id'].isin(devtool_ids)].copy()

        # Group by developer, project, and event type, then get the min/max and total events
        df_dev2tool = (
            df_dev2tool
            .groupby(['developer_id','project_id','event_type'], as_index=False)
            .agg({'first_event': 'min', 'last_event': 'max', 'total_events': 'sum'})
        )

        # Get the weight for each event type
        git_event_weights = self.config.git_event_weights
        df_dev2tool['event_wt'] = df_dev2tool['event_type'].apply(
            lambda evt_type: git_event_weights.get(evt_type, 0.0)
        )

        # Apply a time decay function
        def time_decay2(t):
            if pd.isnull(t) or pd.isnull(ref_time):
                return 1.0
            return event_decay ** ((ref_time - t).total_seconds() / 31536000)
        df_dev2tool['decay'] = df_dev2tool['last_event'].apply(time_decay2)
        df_dev2tool['base_wt'] = df_dev2tool['event_wt'] * df_dev2tool['decay']

        # Get the value v_raw for each developer and project
        df_dev2tool_sum = (
            df_dev2tool
            .groupby(['developer_id','project_id'], as_index=False)['base_wt'].sum()
            .rename(columns={'developer_id':'i','project_id':'j', 'base_wt':'v_raw'})
        )
        df_dev2tool_sum['link_type'] = 'developer_to_devtool'

        #--------------------------------------------------------------------
        # Combine all edges
        #--------------------------------------------------------------------
        
        # Scale the value of v_raw to [0,1] for each edge type
        df_pair_sums['v_scaled'] = self._minmax_scale(df_pair_sums['v_raw'])
        df_onchain_devmax['v_scaled'] = self._minmax_scale(df_onchain_devmax['v_raw'])
        df_dev2tool_sum['v_scaled'] = self._minmax_scale(df_dev2tool_sum['v_raw'])
        
        # Combine all edges
        cols = ['i', 'j', 'link_type', 'v_raw', 'v_scaled']
        df_edges = pd.concat([
            df_pair_sums[cols],
            df_onchain_devmax[cols],
            df_dev2tool_sum[cols],
        ], ignore_index=True)

        # Filter out self-edges
        df_edges = df_edges[df_edges['i'] != df_edges['j']]

        # Multiply by link type weights
        link_type_wts = self.config.link_type_weights
        df_edges['v_final'] = df_edges.apply(
            lambda row: row['v_scaled'] * link_type_wts.get(row['link_type'], 1.0),
            axis=1
        )

        # Store final edges to be used in EigenTrust
        self.analysis['unified_edges'] = df_edges.copy()

    def _run_openrank(self) -> None:
        """Run a single EigenTrust pass on the unified edges."""

        # Use the alpha parameter from config
        alpha = self.config.alpha
        et = EigenTrust(alpha=alpha)

        # Pretrust from onchain projects
        pretrust_scores = self.analysis['onchain_projects_pretrust_scores'].to_dict(orient='records')

        # The edges we want to feed to EigenTrust
        df_edges = self.analysis['unified_edges'].copy()
        
        # Filter out edges with zero or negative weights
        df_edges = df_edges[df_edges['v_final'] > 0]

        # Convert to the format EigenTrust expects: a list of dicts with i, j, v
        edge_records = df_edges.apply(
            lambda row: {'i': row['i'], 'j': row['j'], 'v': row['v_final']},
            axis=1
        ).tolist()

        scores = et.run_eigentrust(edge_records, pretrust_scores)

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
        
        # Replace NaN values with 0.0 without using inplace
        df_results['v'] = df_results['v'].fillna(0.0)

        # Apply eligibility
        df_results['v_aggregated'] = df_results['v'] * df_results['is_eligible']

        # Final normalization across devtooling projects
        total = df_results['v_aggregated'].sum()
        if total > 0:
            df_results['v_aggregated'] = df_results['v_aggregated'] / total

        self.analysis['devtooling_project_results'] = df_results

    def _serialize_results(self) -> None:
        """
        Create a simplified graph of onchain projects -> devtooling projects,
        with edges weighted by the sum of the weights of all edges between them.
        """
        # Get project names and scores
        onchain_names = self.analysis['onchain_projects'].set_index('project_id')[['display_name']]
        dt_results = (
            self.analysis['devtooling_project_results']
            .set_index('project_id')
            .loc[lambda df: df['v_aggregated'] > 0, ['display_name', 'v_aggregated']]
        )
        
        # Get edges
        edges = self.analysis['unified_edges']
        
        # Direct package dependency edges
        direct_edges = (
            edges.query("link_type=='package_dependency'")
            [['i', 'j']]
            .rename(columns={'i': 'onchain_project_id', 'j': 'devtooling_project_id'})
        )

        # Indirect developer edges 
        onchain_dev = (
            edges.query("link_type=='onchain_to_developer'")
            [['i', 'j']]
            .rename(columns={'i': 'onchain_project_id', 'j': 'developer_id'})
        )
        dev_devtool = (
            edges.query("link_type=='developer_to_devtool'")
            [['i', 'j']]
            .rename(columns={'i': 'developer_id', 'j': 'devtooling_project_id'})
        )
        indirect_edges = (
            onchain_dev.merge(dev_devtool, on='developer_id')
            [['onchain_project_id', 'devtooling_project_id']]
        )

        # Combine all edges and merge with project data
        graph = (
            pd.concat([direct_edges, indirect_edges])
            .drop_duplicates()
            .merge(dt_results, left_on='devtooling_project_id', right_index=True)
            .rename(columns={'display_name': 'Devtooling Project'})
            .merge(onchain_names, left_on='onchain_project_id', right_index=True)
            .rename(columns={'display_name': 'Onchain Project'})
        )

        # Calculate scores
        graph['Score'] = (
            graph.groupby('devtooling_project_id')['v_aggregated'].transform('max')
            / graph.groupby('devtooling_project_id')['onchain_project_id'].transform('nunique')
        )

        self.analysis['devtooling_graph'] = graph


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
        alpha=sim.get('alpha', 0.2),
        time_decay=sim.get('time_decay', {}),
        onchain_project_pretrust_weights=sim.get('onchain_project_pretrust_weights', {}),
        dependency_source_weights=sim.get('dependency_source_weights', {}),
        git_event_weights=sim.get('git_event_weights', {}),
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