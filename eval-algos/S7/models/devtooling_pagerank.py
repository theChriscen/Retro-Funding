"""
devtooling.py

Standardized 8-step model pipeline:

1) Instantiate dataclasses with defaults
2) Load YAML config
3) Load raw data
4) Pre-process (graph building, etc.)
5) Run PageRank or weighting algorithm
6) Package into 'analysis'
7) Serialize final results
8) Return 'analysis'
"""

from dataclasses import dataclass, field
import math
import yaml
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


@dataclass
class DataSnapshot:
    """
    Data files for devtooling model. Defaults for local testing.
    """
    data_dir: str = 'eval-algos/S7/data/devtooling_testing'
    onchain_projects: str = 'onchain_projects.csv'
    devtooling_projects: str = 'devtooling_projects.csv'
    project_to_repositories: str = 'project_to_repositories.csv'
    developers_to_repositories: str = 'developers_to_repositories.csv'
    package_links: str = 'package_links.csv'


@dataclass
class SimulationConfig:
    """
    Config controlling weighting for onchain metrics, dev contributions, etc.
    """
    onchain_importance: Dict[str, float] = field(default_factory=lambda: {
        'alpha': 0.4,
        'beta': 0.4,
        'gamma': 0.2
    })
    package_dependency: Dict[str, Any] = field(default_factory=lambda: {
        'base_dependency_weight': 1.0,
        'dependency_source_weights': {
            'CARGO': 2.0,
            'NPM': 1.0,
            'GITHUB': 0.5
        }
    })
    developer_connections: Dict[str, Any] = field(default_factory=lambda: {
        'event_weights': {
            'COMMIT_CODE': 3,
            'FORKED': 1,
            'STARRED': 1,
            'ISSUE_CLOSED': 1,
            'ISSUE_COMMENT': 2,
            'ISSUE_OPENED': 2,
            'ISSUE_REOPENED': 1,
            'PULL_REQUEST_CLOSED': 1,
            'PULL_REQUEST_MERGED': 2,
            'PULL_REQUEST_OPENED': 2,
            'PULL_REQUEST_REOPENED': 1,
            'PULL_REQUEST_REVIEW_COMMENT': 2
        },
        'fanout_exponent': 1.0,
        'time_decay': {
            'half_life_days': 180
        }
    })
    pagerank: Dict[str, Any] = field(default_factory=lambda: {
        'damping_factor': 0.85,
        'max_iterations': 100,
        'convergence_threshold': 1e-6
    })


class DevtoolingCalculator:
    """
    Encapsulates the logic to:
      - compute onchain_importance
      - build Weighted PageRank graph
      - compute additional devtool metrics
      - run final PageRank

    Produces an 'analysis' dict of intermediate DataFrames.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def run_analysis(
        self,
        df_onchain: pd.DataFrame,
        df_devtool: pd.DataFrame,
        df_pkg_links: pd.DataFrame,
        df_proj_to_repos: pd.DataFrame,
        df_devs_repos: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Main pipeline producing an 'analysis' dictionary with all intermediate steps.
        """
        analysis = {}

        # Step 1: Compute onchain importance
        analysis["df_onchain_with_importance"] = self._compute_onchain_importance(df_onchain)

        # Step 2: Join dev data (projects <-> repositories <-> developers)
        onchain_pids = set(analysis["df_onchain_with_importance"]["project_id"].unique())
        devtool_pids = set(df_devtool["project_id"].unique())
        repo_map = self._map_repositories_to_projects(df_proj_to_repos, onchain_pids, devtool_pids)
        analysis["df_dev_joined"] = self._join_developer_data(df_devs_repos, repo_map)

        # Step 3: Build Weighted PageRank graph
        G = self._build_graph(
            analysis["df_onchain_with_importance"],
            df_devtool,
            df_pkg_links,
            analysis["df_dev_joined"]
        )
        analysis["pagerank_graph"] = G

        # Step 4: Run Weighted PageRank
        analysis["pagerank_scores"] = self._compute_page_rank(G)

        # Step 5: Compute additional onchain/devtool metrics for reference
        analysis["df_onchain_full"] = self._compute_onchain_metrics(
            analysis["df_onchain_with_importance"],
            analysis["df_dev_joined"]
        )
        analysis["df_devtool_full"] = self._compute_devtool_metrics(
            df_devtool,
            analysis["df_dev_joined"],
            df_pkg_links
        )

        return analysis

    # --------------------------------------------------------------------
    # Internal methods
    # --------------------------------------------------------------------

    def _compute_onchain_importance(self, df_onchain: pd.DataFrame) -> pd.DataFrame:
        """
        1) Normalize total_transactions, total_active_users, total_gas_fee_contribution
        2) Summarize as onchain_importance = alpha * norm_tx + beta * norm_users + gamma * norm_gas
        """
        alpha = self.config.onchain_importance['alpha']
        beta = self.config.onchain_importance['beta']
        gamma = self.config.onchain_importance['gamma']

        df = df_onchain.copy()
        df['norm_tx'] = self._minmax_scale(df['total_transactions'])
        df['norm_users'] = self._minmax_scale(df['total_active_users'])
        df['norm_gas'] = self._minmax_scale(df['total_gas_fee_contribution'])

        df['onchain_importance'] = (
            alpha * df['norm_tx'] +
            beta  * df['norm_users'] +
            gamma * df['norm_gas']
        )
        return df

    def _map_repositories_to_projects(
        self,
        df_projects_to_repos: pd.DataFrame,
        onchain_pids: set,
        devtool_pids: set
    ) -> Dict[int, Dict[str, str]]:
        """
        Create a map: repo_id -> { 'project_id': int, 'type': 'onchain'|'devtool'|'unknown' }
        """
        repo_map = {}
        for row in df_projects_to_repos.itertuples():
            rid, pid = row.repository_id, row.project_id
            if pid in onchain_pids:
                ptype = 'onchain'
            elif pid in devtool_pids:
                ptype = 'devtool'
            else:
                ptype = 'unknown'
            repo_map[rid] = {'project_id': pid, 'type': ptype}
        return repo_map

    def _join_developer_data(
        self,
        df_devs_repos: pd.DataFrame,
        repo_map: Dict[int, Dict[str, str]]
    ) -> pd.DataFrame:
        """
        For each record in developers_to_repositories, attach onchain_project_id or devtooling_project_id.
        """
        out_rows = []
        for row in df_devs_repos.itertuples():
            info = repo_map.get(row.repository_id, None)
            if not info:
                out_rows.append({
                    'developer_id': row.developer_id,
                    'event_type': row.event_type,
                    'first_event': row.first_event,
                    'total_events': row.total_events,
                    'onchain_project_id': None,
                    'devtooling_project_id': None
                })
                continue

            if info['type'] == 'onchain':
                out_rows.append({
                    'developer_id': row.developer_id,
                    'event_type': row.event_type,
                    'first_event': row.first_event,
                    'total_events': row.total_events,
                    'onchain_project_id': info['project_id'],
                    'devtooling_project_id': None
                })
            elif info['type'] == 'devtool':
                out_rows.append({
                    'developer_id': row.developer_id,
                    'event_type': row.event_type,
                    'first_event': row.first_event,
                    'total_events': row.total_events,
                    'onchain_project_id': None,
                    'devtooling_project_id': info['project_id']
                })
            else:
                out_rows.append({
                    'developer_id': row.developer_id,
                    'event_type': row.event_type,
                    'first_event': row.first_event,
                    'total_events': row.total_events,
                    'onchain_project_id': None,
                    'devtooling_project_id': None
                })
        return pd.DataFrame(out_rows)

    def _build_graph(
        self,
        df_onchain: pd.DataFrame,
        df_devtool: pd.DataFrame,
        df_pkg_links: pd.DataFrame,
        df_dev_joined: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Builds Weighted PageRank graph:
          - nodes = onchain + devtool
          - edges from package dependencies & dev-based logic
        """
        G = nx.DiGraph()

        # Add onchain & devtool nodes
        for row in df_onchain.itertuples():
            G.add_node(row.project_id, node_type='onchain', onchain_importance=row.onchain_importance)
        for row in df_devtool.itertuples():
            G.add_node(row.project_id, node_type='devtool', onchain_importance=0.0)

        # Package dependencies
        self._add_package_dependencies(G, df_pkg_links)

        # Shared developer edges
        self._add_developer_edges(G, df_onchain, df_dev_joined)

        # Developer-based event weighting
        self._add_dev_event_weights(G, df_dev_joined)

        return G

    def _add_package_dependencies(self, G: nx.DiGraph, df_pkg_links: pd.DataFrame) -> None:
        """
        Insert edges onchain->devtool from package dependencies, weighted by config params.
        """
        base_dep = self.config.package_dependency['base_dependency_weight']
        src_weights = self.config.package_dependency['dependency_source_weights']

        df_pkg = df_pkg_links.rename(columns={'onchain_builder_project_id': 'onchain_project_id'})
        for row in df_pkg.itertuples():
            o_node = row.onchain_project_id
            d_node = row.devtooling_project_id
            if o_node not in G or d_node not in G:
                continue

            dep_src = str(getattr(row, 'dependency_source', 'NPM')).upper()
            weight_inc = base_dep * src_weights.get(dep_src, 1.0)

            if not G.has_edge(o_node, d_node):
                G.add_edge(o_node, d_node, weight=0.0)
            G[o_node][d_node]['weight'] += weight_inc

    def _add_developer_edges(
        self,
        G: nx.DiGraph,
        df_onchain: pd.DataFrame,
        df_dev_joined: pd.DataFrame
    ) -> None:
        """
        Edges from onchain->devtool if the same dev is involved.
        Weighted by dev_trust + onchain_importance.
        """
        dev_trust_scores = self._compute_developer_trust_scores(df_dev_joined)
        onchain_map = df_onchain.set_index('project_id')['onchain_importance'].to_dict()
        github_w = self.config.package_dependency['dependency_source_weights'].get('GITHUB', 0.5)

        dev_map = {}
        for row in df_dev_joined.itertuples():
            d_id = row.developer_id
            if pd.isna(d_id):
                continue
            if d_id not in dev_map:
                dev_map[d_id] = {"onchain": set(), "devtool": set(), "trust": dev_trust_scores.get(d_id, 0.0)}
            if not pd.isna(row.onchain_project_id):
                dev_map[d_id]["onchain"].add(row.onchain_project_id)
            if not pd.isna(row.devtooling_project_id):
                dev_map[d_id]["devtool"].add(row.devtooling_project_id)

        shared_edges = {}
        for d_info in dev_map.values():
            d_trust = d_info["trust"]
            for o_pid in d_info["onchain"]:
                oc_imp = onchain_map.get(o_pid, 0.0)
                for dv_pid in d_info["devtool"]:
                    if (o_pid not in G) or (dv_pid not in G):
                        continue
                    key = (o_pid, dv_pid)
                    if key not in shared_edges:
                        shared_edges[key] = {"count": 0, "sum_w": 0.0}
                    shared_edges[key]["count"] += 1
                    shared_edges[key]["sum_w"] += (0.6 * d_trust + 0.4 * oc_imp)

        for (src, dst), val in shared_edges.items():
            if not G.has_edge(src, dst):
                G.add_edge(src, dst, weight=0.0)
            dev_count = val["count"]
            avg_w = val["sum_w"] / dev_count if dev_count > 0 else 0
            G[src][dst]['weight'] += github_w * math.log1p(dev_count) * (1 + avg_w)

    def _add_dev_event_weights(self, G: nx.DiGraph, df_dev_joined: pd.DataFrame) -> None:
        """
        Additional weighting from developer events (time decay, fanout).
        """
        dev_trust_scores = self._compute_developer_trust_scores(df_dev_joined)
        event_w = self.config.developer_connections['event_weights']
        fanout_exp = self.config.developer_connections['fanout_exponent']
        half_life = self.config.developer_connections['time_decay']['half_life_days']
        now = datetime.now()

        dev_to_devtools = {}
        for row in df_dev_joined.itertuples():
            if row.devtooling_project_id is not None:
                dev_to_devtools.setdefault(row.developer_id, set()).add(row.devtooling_project_id)

        edge_sums = {}
        for row in df_dev_joined.itertuples():
            if pd.isna(row.onchain_project_id) or pd.isna(row.devtooling_project_id):
                continue
            if (row.onchain_project_id not in G) or (row.devtooling_project_id not in G):
                continue

            d_id = row.developer_id
            trust = dev_trust_scores.get(d_id, 0.0)
            base_w = event_w.get(str(row.event_type).upper(), 1.0)
            event_val = base_w * (1 + trust)

            # time-decay
            decay = self._time_decay_weight(row.first_event, now, half_life)

            # fanout
            f_size = len(dev_to_devtools.get(d_id, [])) or 1
            partial = event_val * decay * (1.0 / (f_size ** fanout_exp))

            key = (row.onchain_project_id, row.devtooling_project_id)
            edge_sums[key] = edge_sums.get(key, 0.0) + partial

        for (src, dst), val in edge_sums.items():
            if not G.has_edge(src, dst):
                G.add_edge(src, dst, weight=0.0)
            G[src][dst]['weight'] += val

    def _compute_developer_trust_scores(self, df_dev_joined: pd.DataFrame) -> Dict[int, float]:
        """
        Simple dev trust formula, normalized across global maxima:
          trust = 0.4*(#onchain_projects factor) + 0.3*(total_events factor) + 0.3*(commit ratio)
        """
        dev_stats = {}
        for row in df_dev_joined.itertuples():
            d_id = row.developer_id
            if pd.isna(d_id):
                continue
            if d_id not in dev_stats:
                dev_stats[d_id] = {
                    'onchain_projects': set(),
                    'total_events': 0,
                    'commit_events': 0
                }
            st = dev_stats[d_id]
            if not pd.isna(row.onchain_project_id):
                st['onchain_projects'].add(row.onchain_project_id)
            st['total_events'] += row.total_events
            if str(row.event_type).upper() == 'COMMIT_CODE':
                st['commit_events'] += row.total_events

        # max for normalization
        max_onch = max(len(x['onchain_projects']) for x in dev_stats.values()) if dev_stats else 1
        max_events = max(x['total_events'] for x in dev_stats.values()) if dev_stats else 1

        trust_scores = {}
        for d_id, st in dev_stats.items():
            onchain_factor = len(st['onchain_projects']) / max_onch
            event_factor = st['total_events'] / max_events
            commit_ratio = 0.0
            if st['total_events'] > 0:
                commit_ratio = st['commit_events'] / st['total_events']
            trust = 0.4 * onchain_factor + 0.3 * event_factor + 0.3 * commit_ratio
            trust_scores[d_id] = trust
        return trust_scores

    def _time_decay_weight(self, event_date_str: Optional[str], ref_date: datetime, half_life_days: float) -> float:
        """
        Compute time-decay factor = 0.5^(days_since_event / half_life).
        """
        if not isinstance(event_date_str, str):
            return 1.0
        date_str = event_date_str.replace(" UTC", "")
        dt_format = "%Y-%m-%d %H:%M:%S.%f"
        if '.' not in date_str:
            dt_format = "%Y-%m-%d %H:%M:%S"

        try:
            evt_dt = datetime.strptime(date_str, dt_format)
        except ValueError:
            return 1.0

        delta_days = (ref_date - evt_dt).days
        if delta_days < 0:
            delta_days = 0
        return 0.5 ** (delta_days / half_life_days)

    def _compute_page_rank(self, G: nx.DiGraph) -> pd.DataFrame:
        """
        Weighted PageRank with onchain_importance-based personalization.
        Returns DataFrame with columns [project_id, node_type, onchain_importance, pagerank_score].
        """
        damping = self.config.pagerank['damping_factor']
        max_iter = self.config.pagerank['max_iterations']
        conv_thresh = self.config.pagerank['convergence_threshold']

        n = G.number_of_nodes()
        ranks = {node: 1.0 / n for node in G.nodes()}

        # personalization
        sum_onchain = sum(G.nodes[node].get('onchain_importance', 0.0) for node in G.nodes())
        if sum_onchain < 1e-12:
            sum_onchain = 1.0

        personalization = {}
        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'onchain':
                personalization[node] = G.nodes[node]['onchain_importance'] / sum_onchain
            else:
                personalization[node] = 0.0

        iteration = 0
        diff = float('inf')
        while iteration < max_iter and diff > conv_thresh:
            new_ranks = {}
            diff = 0.0

            # out-sums
            out_sum_map = {}
            for u in G.nodes():
                s = sum(e['weight'] for e in G[u].values())
                if s < 1e-12:
                    s = 1e-12
                out_sum_map[u] = s

            # update
            for v in G.nodes():
                rank_sum = (1 - damping) * personalization[v]
                for u in G.predecessors(v):
                    w_uv = G[u][v]['weight']
                    rank_sum += damping * (ranks[u] * w_uv / out_sum_map[u])
                new_ranks[v] = rank_sum

            for node in G.nodes():
                diff += abs(new_ranks[node] - ranks[node])
            ranks = new_ranks
            iteration += 1

        # Build final DataFrame
        rows = []
        for node, score in ranks.items():
            nd = G.nodes[node]
            rows.append({
                'project_id': node,
                'node_type': nd['node_type'],
                'onchain_importance': nd['onchain_importance'],
                'pagerank_score': score
            })
        return pd.DataFrame(rows)

    def _compute_onchain_metrics(self, df_onchain: pd.DataFrame, df_dev_joined: pd.DataFrame) -> pd.DataFrame:
        """
        For each onchain project, compute # distinct devs who performed 'COMMIT_CODE'.
        """
        df_filt = df_dev_joined.dropna(subset=['onchain_project_id']).copy()
        df_filt['event_type'] = df_filt['event_type'].str.upper()
        df_filt = df_filt[df_filt['event_type'] == 'COMMIT_CODE']

        df_count = df_filt.groupby('onchain_project_id')['developer_id'].nunique().reset_index()
        df_count.columns = ['project_id', 'devs_COMMIT_CODE']
        df_out = df_onchain.merge(df_count, on='project_id', how='left')
        df_out['devs_COMMIT_CODE'] = df_out['devs_COMMIT_CODE'].fillna(0).astype(int)
        return df_out

    def _compute_devtool_metrics(
        self,
        df_devtool: pd.DataFrame,
        df_dev_joined: pd.DataFrame,
        df_pkg_links: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For devtools:
          - devs_<EVENT> = # of devs performing <EVENT>
          - onchain_devs_<EVENT> = # devs w/ onchain COMMIT_CODE who also performed <EVENT> on devtool
          - num_onchain_projects_depending
        """
        df_onch_commits = df_dev_joined.dropna(subset=['onchain_project_id']).copy()
        df_onch_commits['event_type'] = df_onch_commits['event_type'].str.upper()
        df_onch_commits = df_onch_commits[df_onch_commits['event_type'] == 'COMMIT_CODE']
        onchain_commit_devs = set(df_onch_commits['developer_id'].unique())

        df_devtools_only = df_dev_joined.dropna(subset=['devtooling_project_id']).copy()
        df_devtools_only['event_type'] = df_devtools_only['event_type'].str.upper()

        # devs_<EVENT>
        df_count_all = (
            df_devtools_only
            .groupby(['devtooling_project_id','event_type'])['developer_id']
            .nunique()
            .reset_index(name='count_devs')
        )
        pivot_all = df_count_all.pivot(
            index='devtooling_project_id',
            columns='event_type',
            values='count_devs'
        ).fillna(0).reset_index()
        evcols = [c for c in pivot_all.columns if c != 'devtooling_project_id']
        pivot_all.rename(
            columns=dict(zip(evcols, [f"devs_{c}" for c in evcols])),
            inplace=True
        )
        pivot_all.rename(columns={'devtooling_project_id': 'project_id'}, inplace=True)

        # onchain_devs_<EVENT>
        df_devtools_only['is_onchain_dev'] = df_devtools_only['developer_id'].isin(onchain_commit_devs)
        df_onchain_subset = df_devtools_only[df_devtools_only['is_onchain_dev']]
        df_count_onchain = (
            df_onchain_subset
            .groupby(['devtooling_project_id','event_type'])['developer_id']
            .nunique()
            .reset_index(name='count_onchain_devs')
        )
        pivot_onchain = df_count_onchain.pivot(
            index='devtooling_project_id',
            columns='event_type',
            values='count_onchain_devs'
        ).fillna(0).reset_index()
        evcols2 = [c for c in pivot_onchain.columns if c != 'devtooling_project_id']
        pivot_onchain.rename(
            columns=dict(zip(evcols2, [f"onchain_devs_{c}" for c in evcols2])),
            inplace=True
        )
        pivot_onchain.rename(columns={'devtooling_project_id': 'project_id'}, inplace=True)

        df_dev_metrics = pivot_all.merge(pivot_onchain, on='project_id', how='outer').fillna(0)

        # package references
        df_pkg = df_pkg_links.rename(columns={'onchain_builder_project_id': 'onchain_project_id'})
        df_refs = (
            df_pkg.groupby('devtooling_project_id')['onchain_project_id']
            .nunique()
            .reset_index(name='num_onchain_projects_depending')
        )
        df_refs.rename(columns={'devtooling_project_id': 'project_id'}, inplace=True)
        df_dev_metrics = df_dev_metrics.merge(df_refs, on='project_id', how='left')
        df_dev_metrics['num_onchain_projects_depending'] = df_dev_metrics['num_onchain_projects_depending'].fillna(0)

        # Merge into df_devtool
        df_out = df_devtool.merge(df_dev_metrics, on='project_id', how='left')
        numeric_cols = df_out.select_dtypes(include=['number']).columns
        df_out[numeric_cols] = df_out[numeric_cols].fillna(0)
        return df_out

    def _minmax_scale(self, series: pd.Series) -> pd.Series:
        """
        Simple min-max scale for numeric columns, fallback to 1.0 if no range.
        """
        if series.min() == series.max():
            return pd.Series([1.0]*len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min())


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
        data_dir=ycfg['data_snapshot'].get('data_dir', 'eval-algos/S7/data/devtooling_testing'),
        onchain_projects=ycfg['data_snapshot'].get('onchain_projects', 'onchain_projects.csv'),
        devtooling_projects=ycfg['data_snapshot'].get('devtooling_projects', 'devtooling_projects.csv'),
        project_to_repositories=ycfg['data_snapshot'].get('project_to_repositories', 'project_to_repositories.csv'),
        developers_to_repositories=ycfg['data_snapshot'].get('developers_to_repositories', 'developers_to_repositories.csv'),
        package_links=ycfg['data_snapshot'].get('package_links', 'package_links.csv')
    )

    sc = SimulationConfig(
        onchain_importance=ycfg['simulation']['onchain_importance'],
        package_dependency=ycfg['simulation']['package_dependency'],
        developer_connections=ycfg['simulation']['developer_connections'],
        pagerank=ycfg['simulation']['pagerank']
    )
    return ds, sc


def load_data(ds: DataSnapshot) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSVs for onchain, devtool, etc. from data_dir.
    """
    def path(fname: str):
        return f"{ds.data_dir}/{fname}"

    df_onchain = pd.read_csv(path(ds.onchain_projects))
    df_devtool = pd.read_csv(path(ds.devtooling_projects))
    df_pkg_links = pd.read_csv(path(ds.package_links))
    df_proj_to_repos = pd.read_csv(path(ds.project_to_repositories))
    df_devs_repos = pd.read_csv(path(ds.developers_to_repositories))
    return df_onchain, df_devtool, df_pkg_links, df_proj_to_repos, df_devs_repos


# ------------------------------------------------------------------------
# Main pipeline entry-point & packaging
# ------------------------------------------------------------------------

def run_simulation(config_path: str) -> Dict[str, Any]:
    """
    Orchestrates the entire pipeline:

      1) Instantiate defaults
      2) load_config
      3) load_data
      4) Pre-process (compute onchain importance, build graph)
      5) Run Weighted PageRank
      6) Assemble 'analysis'
      7) (Optionally) Save
      8) Return 'analysis'
    """
    ds, sim_cfg = load_config(config_path)
    df_onchain, df_devtool, df_pkg_links, df_proj_to_repos, df_devs_repos = load_data(ds)

    # Create DevtoolingCalculator and run analysis
    calculator = DevtoolingCalculator(sim_cfg)
    analysis = calculator.run_analysis(
        df_onchain,
        df_devtool,
        df_pkg_links,
        df_proj_to_repos,
        df_devs_repos
    )

    # Combine final devtool results with pagerank scores
    df_devtool_full = analysis["df_devtool_full"]
    df_pagerank = analysis["pagerank_scores"]

    df_devtool_final = df_devtool_full.merge(
        df_pagerank[df_pagerank['node_type'] == 'devtool'][['project_id', 'pagerank_score']],
        on='project_id',
        how='left'
    ).fillna(0)

    # Sort by pagerank_score for final ranking
    analysis["final_results"] = df_devtool_final.sort_values('pagerank_score', ascending=False)

    # Store snapshot & config references
    analysis["data_snapshot"] = ds
    analysis["simulation_config"] = sim_cfg

    return analysis


# ------------------------------------------------------------------------
# Serialize
# ------------------------------------------------------------------------

def save_results(analysis: Dict[str, Any]) -> None:
    """
    Write final results to a CSV and/or JSON if data_snapshot is available.
    """
    ds = analysis.get("data_snapshot")
    if ds is None:
        print("No DataSnapshot found; skipping file output.")
        return

    out_csv = f"{ds.data_dir}/devtooling_testing_results.csv"
    analysis["final_results"].to_csv(out_csv, index=False)
    print(f"Saved devtool CSV to {out_csv}")

    out_json = f"{ds.data_dir}/devtooling_testing_results.json"
    _generate_combined_json(
        analysis["df_onchain_full"],
        analysis["df_devtool_full"],
        analysis["pagerank_scores"],
        out_json
    )

    print(f"Saved devtool JSON to {out_json}")


def _generate_combined_json(
    df_onchain: pd.DataFrame,
    df_devtool: pd.DataFrame,
    df_pagerank: pd.DataFrame,
    out_json: str
) -> None:
    """
    Helper function for merging devtool/onchain data with pagerank for JSON output.
    """
    df_onchain_merged = df_onchain.merge(
        df_pagerank[df_pagerank["node_type"] == "onchain"][["project_id", "pagerank_score"]],
        on="project_id",
        how="left"
    )
    df_devtool_merged = df_devtool.merge(
        df_pagerank[df_pagerank["node_type"] == "devtool"][["project_id", "pagerank_score"]],
        on="project_id",
        how="left"
    )

    combined = {
        "onchain_projects": df_onchain_merged.to_dict(orient="records"),
        "devtool_projects": df_devtool_merged.to_dict(orient="records")
    }

    import json
    with open(out_json, 'w') as f:
        json.dump(combined, f, indent=2)


# ------------------------------------------------------------------------
# main()
# ------------------------------------------------------------------------

def main():
    """
    Standard entry-point for running this script from CLI.
    """
    # Example config path
    config_path = 'eval-algos/S7/weights/devtooling_testing.yaml'
    analysis = run_simulation(config_path)
    save_results(analysis)


if __name__ == '__main__':
    main()