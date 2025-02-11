from dataclasses import dataclass, field

import argparse
import math
import yaml
import pandas as pd
import networkx as nx
from datetime import datetime
import os
import json
from typing import Dict, Set, List, Tuple, Optional, Any

from utils.constants import DEFAULT_CONFIG, DATA_DIR

@dataclass
class DataSnapshot:
    data_dir: str = 'eval-algos/S7/data/devtooling_testing'
    onchain_projects: str = 'onchain_projects.csv'
    devtooling_projects: str = 'devtooling_projects.csv'
    project_to_repositories: str = 'project_to_repositories.csv'
    developers_to_repositories: str = 'developers_to_repositories.csv'
    package_links: str = 'package_links.csv'
    output_csv: str = 'pagerank_results.csv'
    output_json: str = 'pagerank_results.json'

@dataclass
class SimulationConfig:
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
        }
    })
    pagerank: Dict[str, Any] = field(default_factory=lambda: {
        'damping_factor': 0.85,
        'max_iterations': 100,
        'convergence_threshold': 1e-6
    })

def load_config(config_path: str) -> Tuple[DataSnapshot, SimulationConfig]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Load data snapshot config
    data_snapshot = DataSnapshot(**yaml_config['data_snapshot'])
    
    # Load simulation config
    sim_config = SimulationConfig(
        onchain_importance=yaml_config['simulation']['onchain_importance'],
        package_dependency=yaml_config['simulation']['package_dependency'],
        developer_connections=yaml_config['simulation']['developer_connections'],
        pagerank=yaml_config['simulation']['pagerank']
    )
    
    return data_snapshot, sim_config

def load_data(data_snapshot: DataSnapshot) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required data files using DataSnapshot configuration."""
    find_path = lambda filename: os.path.join(data_snapshot.data_dir, filename)
    
    df_onchain = pd.read_csv(find_path(data_snapshot.onchain_projects))
    df_devtool = pd.read_csv(find_path(data_snapshot.devtooling_projects))
    df_pkg_links = pd.read_csv(find_path(data_snapshot.package_links))
    df_proj_to_repos = pd.read_csv(find_path(data_snapshot.project_to_repositories))
    df_devs_repos = pd.read_csv(find_path(data_snapshot.developers_to_repositories))
    
    return df_onchain, df_devtool, df_pkg_links, df_proj_to_repos, df_devs_repos

def minmax_scale(series: pd.Series) -> pd.Series:
    """Simple min-max scale for numeric columns."""
    if series.min() == series.max():
        return pd.Series([1.0]*len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def compute_onchain_importance(
    df_onchain: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float
) -> pd.DataFrame:
    """
    1) Normalizes total_transactions, total_active_users, total_gas_fee_contribution
    2) onchain_importance = alpha*norm_tx + beta*norm_users + gamma*norm_gas
    """
    df = df_onchain.copy()
    df['norm_tx'] = minmax_scale(df['total_transactions'])
    df['norm_users'] = minmax_scale(df['total_active_users'])
    df['norm_gas'] = minmax_scale(df['total_gas_fee_contribution'])

    df['onchain_importance'] = (
        alpha * df['norm_tx'] +
        beta  * df['norm_users'] +
        gamma * df['norm_gas']
    )
    return df

def time_decay_weight(
    event_date_str: Optional[str],
    ref_date: datetime,
    half_life_days: float
) -> float:
    """Compute time-decay factor = 0.5^(days_since_event / half_life)."""
    if not isinstance(event_date_str, str):
        return 1.0
    clean_str = event_date_str.replace(" UTC","")
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    if '.' not in clean_str:
        date_format = "%Y-%m-%d %H:%M:%S"

    try:
        event_dt = datetime.strptime(clean_str, date_format)
    except:
        return 1.0
    delta_days = (ref_date - event_dt).days
    if delta_days < 0:
        delta_days = 0
    return 0.5 ** (delta_days / half_life_days)

def map_repositories_to_projects(
    df_projects_to_repos: pd.DataFrame,
    onchain_pids: Set[int],
    devtool_pids: Set[int]
) -> Dict[int, Dict[str, Any]]:
    """
    Return repo_map: repo_id -> {project_id, type='onchain'|'devtool'|'unknown'}
    """
    repo_map = {}
    for row in df_projects_to_repos.itertuples():
        rid = row.repository_id
        pid = row.project_id
        if pid in onchain_pids:
            repo_type = 'onchain'
        elif pid in devtool_pids:
            repo_type = 'devtool'
        else:
            repo_type = 'unknown'
        repo_map[rid] = {
            'project_id': pid,
            'type': repo_type
        }
    return repo_map

def join_developer_data(df_devs_repos, repo_map):
    """
    For each record in developers_to_repositories:
      -> "onchain_project_id" if it's an onchain repo
      -> "devtooling_project_id" if it's a devtool repo
    """
    out_rows = []
    for row in df_devs_repos.itertuples(index=False):
        repo_info = repo_map.get(row.repository_id, None)
        if not repo_info:
            out_rows.append({
                'developer_id': row.developer_id,
                'event_type': row.event_type,
                'first_event': row.first_event,
                'total_events': row.total_events,
                'onchain_project_id': None,
                'devtooling_project_id': None
            })
            continue

        if repo_info['type'] == 'onchain':
            onchain_pid = repo_info['project_id']
            devtool_pid = None
        elif repo_info['type'] == 'devtool':
            onchain_pid = None
            devtool_pid = repo_info['project_id']
        else:
            onchain_pid = None
            devtool_pid = None

        out_rows.append({
            'developer_id': row.developer_id,
            'event_type': row.event_type,
            'first_event': row.first_event,
            'total_events': row.total_events,
            'onchain_project_id': onchain_pid,
            'devtooling_project_id': devtool_pid
        })
    return pd.DataFrame(out_rows)

def compute_developer_trust_scores(
    df_dev_repos_join: pd.DataFrame
) -> Dict[int, float]:
    """
    Compute developer trust scores based on:
    1. Number of onchain projects they've contributed to
    2. Total number of contributions
    3. Types of contributions (weighted by event type 'COMMIT_CODE')
    Returns: dict of developer_id -> trust_score
    """
    dev_stats: Dict[int, Dict[str, Any]] = {}
    
    for row in df_dev_repos_join.itertuples():
        if pd.isna(row.developer_id):
            continue
        if row.developer_id not in dev_stats:
            dev_stats[row.developer_id] = {
                'onchain_projects': set(),
                'total_events': 0,
                'commit_events': 0
            }
        stats = dev_stats[row.developer_id]
        # track onchain projects
        if not pd.isna(row.onchain_project_id):
            stats['onchain_projects'].add(row.onchain_project_id)
        # track total events
        stats['total_events'] += row.total_events
        if row.event_type.upper() == 'COMMIT_CODE':
            stats['commit_events'] += row.total_events

    # figure out normalizing factors
    max_onchain = max(len(s['onchain_projects']) for s in dev_stats.values()) if dev_stats else 1
    max_events = max(s['total_events'] for s in dev_stats.values()) if dev_stats else 1

    trust_scores: Dict[int, float] = {}
    for dev_id, stats in dev_stats.items():
        onchain_factor = len(stats['onchain_projects']) / max_onchain
        event_factor = stats['total_events'] / max_events
        commit_ratio = (stats['commit_events'] / stats['total_events']) if stats['total_events'] > 0 else 0
        trust_score = (
            0.4 * onchain_factor +  # Weight for onchain project diversity
            0.3 * event_factor +    # Weight for overall activity
            0.3 * commit_ratio      # Weight for commit ratio
        )
        trust_scores[dev_id] = trust_score

    return trust_scores

def compute_event_weight(
    event_type: str,
    dev_trust_score: float,
    event_weights: Dict[str, float]
) -> float:
    """
    Combine base event weight with developer trust
    """
    base_weight = event_weights.get(event_type.upper(), 1.0)
    return base_weight * (1 + dev_trust_score)

def build_graph(
    config: SimulationConfig,
    df_onchain: pd.DataFrame,
    df_devtool: pd.DataFrame,
    df_package_links: pd.DataFrame,
    df_dev_repos_join: pd.DataFrame
) -> nx.DiGraph:
    """Build a Weighted PageRank graph: node_id=project_id."""
    G = nx.DiGraph()

    # 1) Onchain and devtool nodes
    for row in df_onchain.itertuples():
        G.add_node(row.project_id, node_type='onchain', onchain_importance=row.onchain_importance)
    for row in df_devtool.itertuples():
        G.add_node(row.project_id, node_type='devtool', onchain_importance=0.0)

    # 2) Package dependencies
    df_pkg = df_package_links.rename(columns={'onchain_builder_project_id': 'onchain_project_id'})
    base_dep_w = config.package_dependency['base_dependency_weight']
    source_weights = config.package_dependency['dependency_source_weights']

    for row in df_pkg.itertuples():
        o_node = row.onchain_project_id
        d_node = row.devtooling_project_id
        if (o_node not in G) or (d_node not in G):
            continue
        dep_src = str(row.dependency_source).upper() if hasattr(row, 'dependency_source') else 'NPM'
        src_factor = source_weights.get(dep_src, 1.0)
        weight_inc = base_dep_w * src_factor

        if not G.has_edge(o_node, d_node):
            G.add_edge(o_node, d_node, weight=0.0)
        G[o_node][d_node]['weight'] += weight_inc

    # Prepare map for onchain importance
    onchain_importance_map = df_onchain.set_index('project_id')['onchain_importance'].to_dict()

    # 3) Developer trust
    dev_trust_scores = compute_developer_trust_scores(df_dev_repos_join)

    # 4) Shared developer edges (onchain -> devtool)
    github_weight = source_weights.get('GITHUB', 0.5)
    shared_dev_edges = {}
    dev_map = {}

    for row in df_dev_repos_join.itertuples():
        d_id = row.developer_id
        if pd.isna(d_id):
            continue
        if d_id not in dev_map:
            dev_map[d_id] = {
                'onchain': set(),
                'devtool': set(),
                'trust_score': dev_trust_scores.get(d_id, 0.0)
            }
        if not pd.isna(row.onchain_project_id):
            dev_map[d_id]['onchain'].add(row.onchain_project_id)
        if not pd.isna(row.devtooling_project_id):
            dev_map[d_id]['devtool'].add(row.devtooling_project_id)

    for dev_info in dev_map.values():
        trust_score = dev_info['trust_score']
        for o_pid in dev_info['onchain']:
            oc_imp = onchain_importance_map.get(o_pid, 0.0)
            for d_pid in dev_info['devtool']:
                if (o_pid not in G) or (d_pid not in G):
                    continue
                key = (o_pid, d_pid)
                if key not in shared_dev_edges:
                    shared_dev_edges[key] = {
                        'count': 0,
                        'weighted_sum': 0.0
                    }
                shared_dev_edges[key]['count'] += 1
                # Weighted sum
                shared_dev_edges[key]['weighted_sum'] += (0.6 * trust_score + 0.4 * oc_imp)

    for (src, dst), edge_data in shared_dev_edges.items():
        if not G.has_edge(src, dst):
            G.add_edge(src, dst, weight=0.0)
        dev_count = edge_data['count']
        avg_weight = edge_data['weighted_sum'] / dev_count if dev_count > 0 else 0
        G[src][dst]['weight'] += github_weight * math.log1p(dev_count) * (1 + avg_weight)

    # 5) Developer-based event weighting
    event_weights = config.developer_connections['event_weights']
    fanout_exp = config.developer_connections.get('fanout_exponent', 1.0)
    half_life_days = config.developer_connections.get('time_decay', {}).get('half_life_days', 180)
    now = datetime.now()

    dev_to_devtools = {}
    for row in df_dev_repos_join.itertuples():
        if row.devtooling_project_id is not None:
            dev_to_devtools.setdefault(row.developer_id, set()).add(row.devtooling_project_id)

    edge_sums = {}
    for row in df_dev_repos_join.itertuples():
        if pd.isna(row.onchain_project_id) or pd.isna(row.devtooling_project_id):
            continue
        if (row.onchain_project_id not in G) or (row.devtooling_project_id not in G):
            continue

        dev_id = row.developer_id
        trust_score = dev_trust_scores.get(dev_id, 0.0)
        e_type = row.event_type.upper()
        event_weight = compute_event_weight(e_type, trust_score, event_weights)

        decay = time_decay_weight(row.first_event, now, half_life_days)
        fanout_size = len(dev_to_devtools.get(dev_id, [])) or 1
        partial_w = event_weight * decay * (1.0 / (fanout_size ** fanout_exp))

        edge_sums.setdefault((row.onchain_project_id, row.devtooling_project_id), 0.0)
        edge_sums[(row.onchain_project_id, row.devtooling_project_id)] += partial_w

    for (src, dst), val in edge_sums.items():
        if not G.has_edge(src, dst):
            G.add_edge(src, dst, weight=0.0)
        G[src][dst]['weight'] += val

    return G

def advanced_weighted_pagerank(
    G: nx.DiGraph,
    damping_factor: float,
    max_iterations: int,
    convergence_threshold: float
) -> Dict[int, float]:
    """Weighted PageRank with onchain_importance personalization."""
    ranks = {}
    n = G.number_of_nodes()
    for node in G.nodes():
        ranks[node] = 1.0 / n

    # personalization
    sum_onchain = 0.0
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'onchain':
            sum_onchain += G.nodes[node]['onchain_importance']
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
    while iteration < max_iterations and diff > convergence_threshold:
        new_ranks = {}
        diff = 0.0

        out_sum_map = {}
        for u in G.nodes():
            wsum = sum(edge_data['weight'] for edge_data in G[u].values())
            if wsum < 1e-12:
                wsum = 1e-12
            out_sum_map[u] = wsum

        for v in G.nodes():
            rank_sum = (1 - damping_factor) * personalization[v]
            for u in G.predecessors(v):
                w_uv = G[u][v]['weight']
                rank_sum += damping_factor * (ranks[u] * w_uv / out_sum_map[u])
            new_ranks[v] = rank_sum

        for node in G.nodes():
            diff += abs(new_ranks[node] - ranks[node])
        ranks = new_ranks
        iteration += 1

    return ranks

def compute_onchain_metrics(df_onchain, df_dev_repos_join):
    """
    Add "devs_COMMIT_CODE" for each onchain project:
      - # of distinct devs who performed "COMMIT_CODE" on that onchain project
    Merge it into df_onchain so we can export it in the results CSV for calibration.
    """
    df_filt = df_dev_repos_join.dropna(subset=['onchain_project_id']).copy()
    df_filt['event_type'] = df_filt['event_type'].str.upper()
    df_filt = df_filt[df_filt['event_type'] == 'COMMIT_CODE']

    df_count = df_filt.groupby('onchain_project_id')['developer_id'].nunique().reset_index()
    df_count.columns = ['project_id', 'devs_COMMIT_CODE']

    df_onchain_new = df_onchain.merge(df_count, on='project_id', how='left')
    df_onchain_new['devs_COMMIT_CODE'] = df_onchain_new['devs_COMMIT_CODE'].fillna(0).astype(int)
    return df_onchain_new

def compute_devtool_metrics(df_devtool, df_dev_repos_join, df_package_links):
    """
    For devtools, we want:
      1) devs_<EVENT> = # of devs performing <EVENT> on that devtool
      2) onchain_devs_<EVENT> = # of devs who have "committed code" on ANY onchain project
         and also performed <EVENT> on that devtool
      3) num_onchain_projects_depending
    """
    df_onchain_commits = df_dev_repos_join.dropna(subset=['onchain_project_id']).copy()
    df_onchain_commits['event_type'] = df_onchain_commits['event_type'].str.upper()
    df_onchain_commits = df_onchain_commits[df_onchain_commits['event_type'] == 'COMMIT_CODE']
    onchain_commit_devs = set(df_onchain_commits['developer_id'].unique())

    df_devtools_only = df_dev_repos_join.dropna(subset=['devtooling_project_id']).copy()
    df_devtools_only['event_type'] = df_devtools_only['event_type'].str.upper()

    df_count_all = (
        df_devtools_only
        .groupby(['devtooling_project_id','event_type'])['developer_id']
        .nunique()
        .reset_index(name='count_devs')
    )
    df_pivot_all = df_count_all.pivot(
        index='devtooling_project_id',
        columns='event_type',
        values='count_devs'
    ).fillna(0).reset_index()
    df_pivot_all.columns.name = None
    event_cols = [c for c in df_pivot_all.columns if c != 'devtooling_project_id']
    df_pivot_all.rename(
        columns=dict(zip(event_cols, [f"devs_{c}" for c in event_cols])),
        inplace=True
    )
    df_pivot_all.rename(columns={'devtooling_project_id': 'project_id'}, inplace=True)

    df_devtools_only['is_onchain_dev'] = df_devtools_only['developer_id'].isin(onchain_commit_devs)
    df_onchain_subset = df_devtools_only[df_devtools_only['is_onchain_dev']].copy()
    df_count_onchain_devs = (
        df_onchain_subset
        .groupby(['devtooling_project_id','event_type'])['developer_id']
        .nunique()
        .reset_index(name='count_onchain_devs')
    )

    df_pivot_onchain = df_count_onchain_devs.pivot(
        index='devtooling_project_id',
        columns='event_type',
        values='count_onchain_devs'
    ).fillna(0).reset_index()
    df_pivot_onchain.columns.name = None
    event_cols = [c for c in df_pivot_onchain.columns if c != 'devtooling_project_id']
    df_pivot_onchain.rename(
        columns=dict(zip(event_cols, [f"onchain_devs_{c}" for c in event_cols])),
        inplace=True
    )
    df_pivot_onchain.rename(columns={'devtooling_project_id': 'project_id'}, inplace=True)

    df_dev_metrics = df_pivot_all.merge(df_pivot_onchain, on='project_id', how='outer').fillna(0)

    df_pkg = df_package_links.rename(columns={
        'onchain_builder_project_id': 'onchain_project_id'
    })
    df_refs = (
        df_pkg.groupby('devtooling_project_id')['onchain_project_id']
        .nunique()
        .reset_index(name='num_onchain_projects_depending')
    )
    df_refs.rename(columns={'devtooling_project_id': 'project_id'}, inplace=True)

    df_dev_metrics = df_dev_metrics.merge(df_refs, on='project_id', how='left')
    df_dev_metrics['num_onchain_projects_depending'] = df_dev_metrics['num_onchain_projects_depending'].fillna(0)

    df_devtool_metrics = df_devtool.merge(df_dev_metrics, on='project_id', how='left')
    numeric_cols = df_devtool_metrics.select_dtypes(include=['number']).columns
    df_devtool_metrics[numeric_cols] = df_devtool_metrics[numeric_cols].fillna(0)
    return df_devtool_metrics

def merge_pagerank_scores(
    df_base: pd.DataFrame,
    df_pagerank: pd.DataFrame,
    project_type: str
) -> pd.DataFrame:
    """Helper to merge pagerank scores with base dataframe."""
    return df_base.merge(
        df_pagerank[df_pagerank['node_type'] == project_type][['project_id', 'pagerank_score']],
        on='project_id',
        how='left'
    )

def create_pagerank_results(pagerank_scores, G):
    """Convert pagerank scores to DataFrame."""
    rows = []
    for node, score in pagerank_scores.items():
        rows.append({
            'project_id': node,
            'node_type': G.nodes[node]['node_type'],
            'onchain_importance': G.nodes[node]['onchain_importance'],
            'pagerank_score': score
        })
    return pd.DataFrame(rows)

def generate_combined_json(df_onchain_full, df_devtool_metrics_full, df_pagerank, output_json_path):
    """
    Produce a JSON structure with onchain and devtool projects, including pagerank scores.
    """
    df_onchain_merged = merge_pagerank_scores(df_onchain_full, df_pagerank, 'onchain')
    df_devtool_merged = merge_pagerank_scores(df_devtool_metrics_full, df_pagerank, 'devtool')

    output_dict = {
        "onchain_projects": df_onchain_merged.to_dict(orient='records'),
        "devtool_projects": df_devtool_merged.to_dict(orient='records')
    }

    with open(output_json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)
    print(f"Saved combined JSON to {output_json_path}")

def save_results(
    data_snapshot: DataSnapshot,
    df_devtool_final: pd.DataFrame,
    df_onchain_full: pd.DataFrame,
    df_devtool_metrics_full: pd.DataFrame,
    df_pagerank: pd.DataFrame
) -> None:
    """Save final results to CSV and JSON files."""
    find_path = lambda filename: os.path.join(data_snapshot.data_dir, filename)
    
    # Save CSV results
    df_devtool_final.sort_values(by='pagerank_score', ascending=False).to_csv(
        find_path(data_snapshot.output_csv),
        index=False
    )
    
    # Save JSON results
    generate_combined_json(
        df_onchain_full,
        df_devtool_metrics_full,
        df_pagerank,
        find_path(data_snapshot.output_json)
    )
    print("Done. Files saved.")

def main():
    # 0) Initialize configuration
    YAML_FILE = 'eval-algos/S7/weights/devtooling_pagerank.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=YAML_FILE)
    args = parser.parse_args()
    
    data_snapshot, config = load_config(args.config)  # Now returns both configs

    # 1) Load data
    df_onchain, df_devtool, df_pkg_links, df_proj_to_repos, df_devs_repos = load_data(data_snapshot)

    # 2) Map repos to projects
    onchain_pids = set(df_onchain['project_id'].unique())
    devtool_pids = set(df_devtool['project_id'].unique())
    repo_map = map_repositories_to_projects(df_proj_to_repos, onchain_pids, devtool_pids)
    df_dev_repos_join = join_developer_data(df_devs_repos, repo_map)

    # 3) Onchain importance
    alpha = float(config.onchain_importance['alpha'])
    beta  = float(config.onchain_importance['beta'])
    gamma = float(config.onchain_importance['gamma'])
    df_onchain = compute_onchain_importance(df_onchain, alpha, beta, gamma)

    # 4) Build Weighted PageRank graph
    G = build_graph(config, df_onchain, df_devtool, df_pkg_links, df_dev_repos_join)

    # 5) Run Weighted PageRank
    d_factor = float(config.pagerank['damping_factor'])
    max_iter = int(config.pagerank['max_iterations'])
    conv_thr = float(config.pagerank['convergence_threshold'])
    pagerank_scores = advanced_weighted_pagerank(G, d_factor, max_iter, conv_thr)

    # 6) Create PageRank results DataFrame
    df_pagerank = create_pagerank_results(pagerank_scores, G)

    # 7) Compute extra onchain/devtool metrics
    df_onchain_full = compute_onchain_metrics(df_onchain, df_dev_repos_join)
    df_devtool_metrics_full = compute_devtool_metrics(df_devtool, df_dev_repos_join, df_pkg_links)

    # 8) Save results
    df_devtool_final = merge_pagerank_scores(df_devtool_metrics_full, df_pagerank, 'devtool')
    save_results(
        data_snapshot,
        df_devtool_final,
        df_onchain_full,
        df_devtool_metrics_full,
        df_pagerank
    )

if __name__ == '__main__':
    main()