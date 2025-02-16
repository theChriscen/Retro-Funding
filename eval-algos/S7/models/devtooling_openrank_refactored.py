from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import yaml
import warnings

from openrank_sdk import EigenTrust
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
    devtooling_project_pretrust_weights: Dict[str, float]
    dependency_source_weights: Dict[str, float]
    git_event_weights: Dict[str, float]
    link_type_weights: Dict[str, float]
    eligibility_thresholds: Dict[str, int]


class DevtoolingCalculator:
    """
    Encapsulates logic for generating OpenRank graphs and metrics
    around devtooling projects and their onchain usage and developer links.
    Produces an 'analysis' dict of all intermediate DataFrames.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.analysis = {}

    # --------------------------------------------------------------------
    # Main pipeline (entry to final 'analysis' outputs)
    # --------------------------------------------------------------------

    def run_analysis(
        self,
        df_onchain_projects: pd.DataFrame,
        df_devtooling_projects: pd.DataFrame,
        df_package_links: pd.DataFrame,
        df_developers_to_repositories: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
    
        # 1: Store raw frames
        self.analysis = {
            'onchain_projects': df_onchain_projects,
            'devtooling_projects': df_devtooling_projects,
            'package_links': df_package_links,
            'developers_to_repositories': df_developers_to_repositories
        }

        # Step 1: Seed Onchain Projects with Economic Pretrust
        self._compute_project_pretrust()

        # Step 2: Compute Developer Reputation from Onchain Projects
        self._compute_developer_reputation()

        # Step 3: Aggregate Developer Contributions to Devtooling Projects
        self._aggregate_developer_contributions()

        # Step 4: Weight Package Dependencies
        self._weight_package_dependencies()

        # Step 5: Build the Trust Graph
        self._build_trust_graph()

        # Step 6: Apply EigenTrust
        self._apply_eigentrust()

        # Step 7: Rank and Evaluate Projects
        self._rank_and_evaluate_projects()

        # 8: Serialize final results and detailed graph
        self._serialize_results()
        self._serialize_detailed_graph()

        return self.analysis

    # --------------------------------------------------------------------
    # (Steps 1 - 3 remain unchanged)
    # --------------------------------------------------------------------
    def _compute_project_pretrust(self) -> None:
        # ... [existing code remains unchanged]
        df_onchain = self.analysis['onchain_projects'].copy()
        df_onchain.rename(columns={'project_id': 'i'}, inplace=True)
        wts = self.config.onchain_project_pretrust_weights
        df_onchain['v'] = 0.0
        for col, weight in wts.items():
            if col in df_onchain.columns:
                df_onchain[col] = self._minmax_scale(np.log1p(df_onchain[col]))
                df_onchain['v'] += df_onchain[col] * weight
        onchain_total = df_onchain['v'].sum()
        df_onchain['v'] /= onchain_total

        df_devtooling = self.analysis['devtooling_projects'].copy()
        df_devtooling.rename(columns={'project_id': 'i'}, inplace=True)
        wts = self.config.devtooling_project_pretrust_weights
        df_devtooling['v'] = 0.0
        for col, weight in wts.items():
            if col in df_devtooling.columns:
                df_devtooling[col] = self._minmax_scale(np.log1p(df_devtooling[col]))
                df_devtooling['v'] += df_devtooling[col] * weight
        devtooling_total = df_devtooling['v'].sum()
        df_devtooling['v'] /= devtooling_total

        self.analysis['onchain_projects_pretrust_scores'] = df_onchain
        self.analysis['devtooling_projects_pretrust_scores'] = df_devtooling
        
        df_onchain.sort_values(by='v', ascending=False, inplace=True)
        df_devtooling.sort_values(by='v', ascending=False, inplace=True)

    def _compute_developer_reputation(self) -> None:
        # ... [existing code remains unchanged]
        df_devrepo = self.analysis['developers_to_repositories'].copy()
        onchain_ids = set(self.analysis['onchain_projects']['project_id'].unique())
        df_onchain_dev = df_devrepo[
            (df_devrepo['project_id'].isin(onchain_ids)) &
            (df_devrepo['event_type'] == 'COMMIT_CODE')
        ].copy()
        df_onchain_dev = df_onchain_dev[
            (df_onchain_dev['total_events'] >= 10)
            & ((df_onchain_dev['last_event'] - df_onchain_dev['first_event']) >= pd.Timedelta(days=90))
            #& (df_onchain_dev['last_event'] >= pd.Timestamp.now() - pd.Timedelta(days=180))
        ]
        trusted_developers = df_onchain_dev['developer_id'].unique()
        final_dev_rep = pd.DataFrame({
            'developer_id': trusted_developers,
            'developer_reputation': 1.0 / len(trusted_developers) if len(trusted_developers) > 0 else 0.0
        })
        self.analysis['df_trusted_devs'] = final_dev_rep

    def _aggregate_developer_contributions(self) -> None:
        # ... [existing code remains unchanged except for our later reuse]
        df_devrepo = self.analysis['developers_to_repositories'].copy()
        devtooling_ids = set(self.analysis['devtooling_projects']['project_id'].unique())
        onchain_ids = set(self.analysis['onchain_projects']['project_id'].unique())
        trusted_devs = set(self.analysis['df_trusted_devs']['developer_id'].unique())
        overlapping_projects = onchain_ids.intersection(devtooling_ids)
        df_onchain_dev = df_devrepo[
            (df_devrepo['project_id'].isin(onchain_ids)) &
            (df_devrepo['event_type'] == 'COMMIT_CODE')
        ]
        dev_to_onchain_projects = df_onchain_dev.groupby('developer_id')['project_id'].apply(set).to_dict()
        df_dev2tool = df_devrepo[
            (df_devrepo['project_id'].isin(devtooling_ids)) &
            (df_devrepo['developer_id'].isin(trusted_devs))
        ].copy()
        def is_self_referential(row):
            dev_id = row['developer_id']
            project_id = row['project_id']
            if project_id not in overlapping_projects:
                return False
            if dev_id not in dev_to_onchain_projects:
                return False
            return project_id in dev_to_onchain_projects[dev_id]
        df_dev2tool['is_self_referential'] = df_dev2tool.apply(is_self_referential, axis=1)
        df_dev2tool = df_dev2tool[~df_dev2tool['is_self_referential']]
        ref_time = df_dev2tool['last_event'].max()
        event_decay = self.config.time_decay.get('event_to_devtooling_repo', 1.0)
        git_event_weights = self.config.git_event_weights
        def time_decay_fn(t):
            if pd.isnull(t) or pd.isnull(ref_time):
                return 1.0
            years_diff = (ref_time - t).days / 365.0
            return event_decay ** years_diff
        df_dev2tool['base_wt'] = df_dev2tool.apply(
            lambda row: git_event_weights.get(row['event_type'], 0.0) * time_decay_fn(row['last_event']),
            axis=1
        )
        df_dev2tool_agg = df_dev2tool.groupby(
            ['developer_id', 'project_id'], as_index=False
        )['base_wt'].sum()
        df_dev2tool_agg['sum_wt_by_dev'] = df_dev2tool_agg.groupby('developer_id')['base_wt'].transform('sum')
        df_dev2tool_agg['normalized_contribution'] = 0.0
        mask = df_dev2tool_agg['sum_wt_by_dev'] > 0
        df_dev2tool_agg.loc[mask, 'normalized_contribution'] = (
            df_dev2tool_agg.loc[mask, 'base_wt'] / df_dev2tool_agg.loc[mask, 'sum_wt_by_dev']
        )
        self.analysis['developer_contributions'] = df_dev2tool_agg

    # --------------------------------------------------------------------
    # Step 4: Weight Package Dependencies (Modified)
    # --------------------------------------------------------------------

    def _weight_package_dependencies(self) -> None:
        """
        Weight package dependencies based on their source.
        For each (onchain builder, devtooling) project pair:
          - Find all dependency sources between them
          - Get the weight for each source from config
          - Choose the row with the highest source weight (thereby preserving the dependency_source)
          - Format for graph structure
        """
        df_pkg = self.analysis['package_links'].copy()
        ds_wts = self.config.dependency_source_weights

        # Compute weight for each dependency source
        df_pkg['source_weight'] = df_pkg['dependency_source'].apply(
            lambda ds: ds_wts.get(ds, 1.0)
        )
        # For each (onchain_builder_project_id, devtooling_project_id) pair, keep the row with max weight
        df_pkg = df_pkg.sort_values('source_weight', ascending=False)
        df_pkg_agg = df_pkg.groupby(
            ['onchain_builder_project_id', 'devtooling_project_id'], as_index=False
        ).first()

        df_pkg_agg = df_pkg_agg.rename(columns={
            'onchain_builder_project_id': 'i',
            'devtooling_project_id': 'j'
        })
        df_pkg_agg['v_raw'] = df_pkg_agg['source_weight']
        df_pkg_agg['link_type'] = 'package_dependency'
        # We keep the dependency_source for later use when building the detailed graph.
        self.analysis['package_dependencies'] = df_pkg_agg

    # --------------------------------------------------------------------
    # Step 5: Build the Trust Graph (unchanged)
    # --------------------------------------------------------------------

    def _build_trust_graph(self) -> None:
        # ... [existing code remains unchanged]
        df_dev_contrib = self.analysis['developer_contributions'].copy()
        df_trusted = self.analysis['df_trusted_devs'].copy()
        df_dev_edges = df_dev_contrib.merge(
            df_trusted[['developer_id', 'developer_reputation']],
            on='developer_id',
            how='left'
        )
        df_dev_edges['v_raw'] = df_dev_edges['developer_reputation'] * df_dev_edges['normalized_contribution']
        df_dev_edges = df_dev_edges.rename(columns={
            'developer_id': 'i',
            'project_id': 'j'
        })
        df_dev_edges['link_type'] = 'developer_to_devtool'
        df_pkg = self.analysis['package_dependencies'].copy()
        df_edges = pd.concat([
            df_dev_edges[['i', 'j', 'link_type', 'v_raw']],
            df_pkg[['i', 'j', 'link_type', 'v_raw']]
        ], ignore_index=True)
        link_type_wts = self.config.link_type_weights
        final_frames = []
        for lt, subdf in df_edges.groupby('link_type'):
            subdf = subdf.copy()
            subdf['v_scaled'] = self._minmax_scale(subdf['v_raw'])
            subdf['v_final'] = subdf['v_scaled'] * link_type_wts.get(lt, 1.0)
            final_frames.append(subdf)
        df_edges_final = pd.concat(final_frames, ignore_index=True)
        self.analysis['unified_edges'] = df_edges_final

    # --------------------------------------------------------------------
    # Step 6: Apply EigenTrust (unchanged)
    # --------------------------------------------------------------------

    def _apply_eigentrust(self) -> None:
        # ... [existing code remains unchanged]
        alpha = self.config.alpha
        et = EigenTrust(alpha=alpha)
        pretrust_scores = self.analysis['devtooling_projects_pretrust_scores'].to_dict(orient='records')
        df_edges = self.analysis['unified_edges'].copy()
        df_edges = df_edges[df_edges['v_final'] > 0]
        edge_records = df_edges.apply(
            lambda row: {'i': row['i'], 'j': row['j'], 'v': row['v_final']},
            axis=1
        ).tolist()
        scores = et.run_eigentrust(edge_records, pretrust_scores)
        df_scores = pd.DataFrame(scores, columns=['i', 'v']).set_index('i')
        self.analysis['project_openrank_scores'] = df_scores

    # --------------------------------------------------------------------
    # Step 7: Rank and Evaluate Projects (unchanged)
    # --------------------------------------------------------------------

    def _rank_and_evaluate_projects(self) -> None:
        # ... [existing code remains unchanged]
        df_devtooling = self.analysis['devtooling_projects'].copy()
        df_edges = self.analysis['unified_edges']
        df_scores = self.analysis['project_openrank_scores'].copy()
        df_results = df_devtooling.merge(
            df_scores, left_on='project_id', right_on='i', how='left'
        )
        df_results['v'] = df_results['v'].fillna(0.0)
        df_pkg = df_edges[df_edges['link_type'] == 'package_dependency']
        df_results['total_dependents'] = df_results['project_id'].apply(
            lambda pid: df_pkg[df_pkg['j'] == pid]['i'].nunique()
        )
        df_devtodevtool = df_edges[df_edges['link_type'] == 'developer_to_devtool']
        df_results['developer_links'] = df_results['project_id'].apply(
            lambda pid: df_devtodevtool[df_devtodevtool['j'] == pid]['i'].nunique()
        )
        elig = self.config.eligibility_thresholds
        df_results['is_eligible'] = 0
        df_results.loc[
            (df_results['total_dependents'] >= elig['num_projects_with_package_links']) |
            (df_results['developer_links'] >= elig['num_projects_with_dev_links']),
            'is_eligible'
        ] = 1
        df_results['v_aggregated'] = df_results['v'] * df_results['is_eligible']
        total = df_results['v_aggregated'].sum()
        if total > 0:
            df_results['v_aggregated'] /= total
        self.analysis['devtooling_project_results'] = df_results

    # --------------------------------------------------------------------
    # Step 8: Serialize Detailed Graph with Full Relationship Info (Modified)
    # --------------------------------------------------------------------

    def _serialize_detailed_graph(self) -> None:
        """
        Build a final detailed graph DataFrame with the following columns:
          - date
          - to_id
          - from_id
          - to_display_name
          - from_display_name
          - link_type
          - event_type

        For developer interactions (both developer_to_devtool and developer_to_onchain),
        all events from trusted developers are included. For each (developer, project) group,
        the event that occurs at the earliest timestamp will have its event_type prefixed with "FIRST_"
        and the event at the latest timestamp will have its event_type prefixed with "LAST_".
        (Other events retain their original event_type.)

        For package dependency events:
          - Set date to the current timestamp
          - Derive event_type from the dependency_source (e.g., "NPM_DEPENDENCY")
        """
        # ---------------------------
        # Developer-to-Devtool Events
        # ---------------------------
        df_devs2repos = self.analysis['developers_to_repositories'].copy()
        trusted_devs = set(self.analysis['df_trusted_devs']['developer_id'].unique())
        devtooling_ids = set(self.analysis['devtooling_projects']['project_id'].unique())
        onchain_ids = set(self.analysis['onchain_projects']['project_id'].unique())
        
        # Filter for events on devtooling projects by trusted developers.
        df_devtool = df_devs2repos[
            (df_devs2repos['project_id'].isin(devtooling_ids)) &
            (df_devs2repos['developer_id'].isin(trusted_devs))
        ].copy()
        
        # (Optional) Remove self-referential events if a developer’s commit appears on both onchain and devtooling
        df_onchain_dev = df_devs2repos[
            (df_devs2repos['project_id'].isin(onchain_ids)) &
            (df_devs2repos['event_type'] == 'COMMIT_CODE')
        ]
        dev_to_onchain_projects = df_onchain_dev.groupby('developer_id')['project_id'].apply(set).to_dict()
        def is_self_referential(row):
            dev_id = row['developer_id']
            project_id = row['project_id']
            if project_id not in onchain_ids.intersection(devtooling_ids):
                return False
            if dev_id not in dev_to_onchain_projects:
                return False
            return project_id in dev_to_onchain_projects[dev_id]
        df_devtool['is_self_referential'] = df_devtool.apply(is_self_referential, axis=1)
        df_devtool = df_devtool[~df_devtool['is_self_referential']].copy()
        
        # Choose a single timestamp per event.
        # (Assumption: each row is an individual event, and we use the 'first_event' timestamp.)
        df_devtool['date'] = df_devtool['first_event']
        
        # For each (developer, project) group, compute the min and max event date.
        group_cols = ['developer_id', 'project_id']
        df_devtool['min_date'] = df_devtool.groupby(group_cols)['date'].transform('min')
        df_devtool['max_date'] = df_devtool.groupby(group_cols)['date'].transform('max')
        
        # Create a new column for the modified event type.
        # If an event’s date equals the min, prefix with "FIRST_".
        # If it equals the max, prefix with "LAST_".
        # Otherwise, leave it unchanged.
        df_devtool['modified_event_type'] = df_devtool['event_type']
        df_devtool.loc[df_devtool['date'] == df_devtool['min_date'], 'modified_event_type'] = \
            "FIRST_" + df_devtool['event_type'].astype(str)
        df_devtool.loc[df_devtool['date'] == df_devtool['max_date'], 'modified_event_type'] = \
            "LAST_" + df_devtool['event_type'].astype(str)
        
        # Rename columns to final schema: developer becomes "from_id", project becomes "to_id".
        df_devtool_edges = df_devtool[['date', 'project_id', 'developer_id', 'modified_event_type']].copy()
        df_devtool_edges = df_devtool_edges.rename(columns={
            'project_id': 'to_id',
            'developer_id': 'from_id',
            'modified_event_type': 'event_type'
        })
        df_devtool_edges['link_type'] = 'developer_to_devtool'
        
        
        # -------------------------
        # Developer-to-Onchain Events
        # -------------------------
        df_onchain = self.analysis['developers_to_repositories'].copy()
        df_onchain = df_onchain[
            (df_onchain['project_id'].isin(onchain_ids)) &
            (df_onchain['event_type'] == 'COMMIT_CODE')
        ].copy()
        df_onchain['date'] = df_onchain['first_event']
        group_cols_onchain = ['developer_id', 'project_id']
        df_onchain['min_date'] = df_onchain.groupby(group_cols_onchain)['date'].transform('min')
        df_onchain['max_date'] = df_onchain.groupby(group_cols_onchain)['date'].transform('max')
        df_onchain['modified_event_type'] = df_onchain['event_type']
        df_onchain.loc[df_onchain['date'] == df_onchain['min_date'], 'modified_event_type'] = \
            "FIRST_" + df_onchain['event_type'].astype(str)
        df_onchain.loc[df_onchain['date'] == df_onchain['max_date'], 'modified_event_type'] = \
            "LAST_" + df_onchain['event_type'].astype(str)
        df_onchain_edges = df_onchain[['date', 'project_id', 'developer_id', 'modified_event_type']].copy()
        df_onchain_edges = df_onchain_edges.rename(columns={
            'project_id': 'to_id',
            'developer_id': 'from_id',
            'modified_event_type': 'event_type'
        })
        df_onchain_edges['link_type'] = 'developer_to_onchain'
        
        
        # -------------------------
        # Package Dependency Events
        # -------------------------
        df_pkg = self.analysis['package_dependencies'].copy()
        now_ts = pd.Timestamp.now()
        df_pkg['date'] = now_ts
        df_pkg['event_type'] = df_pkg['dependency_source'].apply(lambda ds: f"{ds.upper()}_DEPENDENCY")
        df_pkg['from_id'] = df_pkg['i']
        df_pkg['to_id'] = df_pkg['j']
        df_pkg['link_type'] = 'package_dependency'
        df_pkg_edges = df_pkg[['date', 'to_id', 'from_id', 'link_type', 'event_type']]
        
        
        # -------------------------
        # Combine All Edges and Add Display Names
        # -------------------------
        detailed_graph = pd.concat([df_devtool_edges, df_onchain_edges, df_pkg_edges],
                                   ignore_index=True)
        
        dt_ids = self.analysis['devtooling_projects'].set_index('project_id')['display_name'].to_dict()
        oc_ids = self.analysis['onchain_projects'].set_index('project_id')['display_name'].to_dict()
        dev_ids = self.analysis['developers_to_repositories'].set_index('developer_id')['developer_name'].to_dict()
        all_ids = {**dt_ids, **oc_ids, **dev_ids}
        
        detailed_graph['to_display_name'] = detailed_graph['to_id'].map(all_ids)
        detailed_graph['from_display_name'] = detailed_graph['from_id'].map(all_ids)
        
        detailed_graph = detailed_graph[['date', 'to_id', 'from_id', 'to_display_name', 
                                           'from_display_name', 'link_type', 'event_type']]
        
        self.analysis['detailed_graph'] = detailed_graph

    # --------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------

    @staticmethod
    def _minmax_scale(values: pd.Series) -> pd.Series:
        vmin, vmax = values.min(), values.max()
        if vmax == vmin:
            return pd.Series([0.5] * len(values), index=values.index)
        return (values - vmin) / (vmax - vmin)
    
    def _serialize_results(self) -> None:
        df_results = self.analysis['devtooling_project_results'].copy()
        df_results.sort_values(by='v_aggregated', ascending=False, inplace=True)
        self.analysis['final_results'] = df_results

# ------------------------------------------------------------------------
# Helpers for loading config & data (unchanged)
# ------------------------------------------------------------------------

def load_config(config_path: str) -> Tuple[DataSnapshot, SimulationConfig]:
    with open(config_path, 'r') as f:
        ycfg = yaml.safe_load(f)

    ds = DataSnapshot(
        data_dir=ycfg['data_snapshot'].get('data_dir', "eval-algos/S7/data/devtooling_testing2"),
        onchain_projects_file=ycfg['data_snapshot'].get('onchain_projects_file', "onchain_projects.csv"),
        devtooling_projects_file=ycfg['data_snapshot'].get('devtooling_projects_file', "devtooling_projects.csv"),
        package_links_file=ycfg['data_snapshot'].get('package_links_file', "package_links.csv"),
        developers_to_repositories_file=ycfg['data_snapshot'].get('developers_to_repositories_file', "developers_to_repositories.csv")
    )

    sim = ycfg.get('simulation', {})
    sc = SimulationConfig(
        alpha=sim.get('alpha', 0.2),
        time_decay=sim.get('time_decay', {}),
        onchain_project_pretrust_weights=sim.get('onchain_project_pretrust_weights', {}),
        devtooling_project_pretrust_weights=sim.get('devtooling_project_pretrust_weights', {}),
        dependency_source_weights=sim.get('dependency_source_weights', {}),
        git_event_weights=sim.get('git_event_weights', {}),
        link_type_weights=sim.get('link_type_weights', {}),
        eligibility_thresholds=sim.get('eligibility_thresholds', {})
    )

    return ds, sc


def load_data(ds: DataSnapshot) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def path(x: str) -> str:
        return f"{ds.data_dir}/{x}"

    df_onchain_projects = pd.read_csv(path(ds.onchain_projects_file))
    df_devtooling_projects = pd.read_csv(path(ds.devtooling_projects_file))
    df_package_links = pd.read_csv(path(ds.package_links_file))

    df_devs2repos = pd.read_csv(path(ds.developers_to_repositories_file))
    df_devs2repos['first_event'] = pd.to_datetime(df_devs2repos['first_event'])
    df_devs2repos['last_event'] = pd.to_datetime(df_devs2repos['last_event'])

    return (
        df_onchain_projects,
        df_devtooling_projects,
        df_package_links,
        df_devs2repos
    )


def run_simulation(config_path: str) -> Dict[str, Any]:
    ds, sim_cfg = load_config(config_path)
    data = load_data(ds)
    calculator = DevtoolingCalculator(sim_cfg)
    analysis = calculator.run_analysis(*data)
    analysis["data_snapshot"] = ds
    analysis["simulation_config"] = sim_cfg
    return analysis


def save_results(analysis: Dict[str, Any]) -> None:
    ds = analysis.get("data_snapshot")
    if ds is None:
        print("No DataSnapshot found; skipping file output.")
        return

    out_path = f"{ds.data_dir}/devtooling_openrank_results.csv"
    final = analysis.get("final_results", None)
    if final is not None:
        final.to_csv(out_path, index=False)
        print(f"[INFO] Saved devtooling openrank results to {out_path}")
    else:
        print("[WARN] No 'final_results' to serialize.")

    detailed_path = f"{ds.data_dir}/detailed_devtooling_graph.csv"
    detailed = analysis.get("detailed_graph", None)
    if detailed is not None:
        detailed.to_csv(detailed_path, index=False)
        print(f"[INFO] Saved detailed graph to {detailed_path}")
    else:
        print("[WARN] No 'detailed_graph' to serialize.")


def main():
    config_path = 'eval-algos/S7/weights/devtooling_openrank_testing.yaml'
    analysis = run_simulation(config_path)
    save_results(analysis)


if __name__ == "__main__":
    main()