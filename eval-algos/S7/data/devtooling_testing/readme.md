# Weighted PageRank for Devtooling Project Importance - Testing Algorithm

## Overview

This algorithm ranks **devtooling projects** by integrating:

1. **Onchain Project Importance**:  
   - Derived from onchain metrics (total transactions, active users, gas contribution).  
   - Used as the [PageRank **personalization** vector](https://en.wikipedia.org/wiki/PageRank#Personalization).

2. **Package Dependencies**:  
   - Onchain -> Devtool edges with a configurable base weight and different multipliers for each dependency source (Cargo, NPM, GitHub, etc.).

3. **Developer Connections**:  
   - Onchain developers who also contribute to or star/fork devtooling projects.  
   - Weighted by developer "trust" (which depends on how active the developer is on onchain projects) plus **time decay** (older actions count less) and **fanout** penalty (a developer who fans out to many devtools has less weight for each).

The result is a **directed, weighted graph** of projects. **PageRank** (with damping factor) is computed over this graph, giving a final `pagerank_score` for each node (particularly for devtooling nodes).

---

## Data & Configuration

1. **YAML File** (e.g. `devtooling_pagerank.yaml`):
   - **`onchain_importance.alpha, beta, gamma`**: Weights for onchain metrics.  
   - **`package_dependency.base_dependency_weight`**: The baseline link weight for a package import.  
   - **`package_dependency.dependency_source_weights`**: Additional multiplier by source (Cargo, NPM, etc.).  
   - **`developer_connections.event_weights`**: The base weight for events like `COMMIT_CODE`, `FORKED`, etc.  
   - **`developer_connections.default_reputation`**: Fallback for developer's trust.  
   - **`developer_connections.fanout_exponent`**: Exponent for penalizing devs who engage with many devtools.  
   - **`developer_connections.time_decay.half_life_days`**: Half-life for older events.  
   - **`pagerank`**: PageRank damping factor, iteration count, and convergence threshold.  
   - **`data_files`**: CSV file paths and output paths.

2. **CSV Inputs**:
   - **`onchain_projects.csv`**: Columns `[project_id, project_name, display_name, total_transactions, total_active_users, total_gas_fee_contribution, ...]`  
   - **`devtooling_projects.csv`**: Columns `[project_id, project_name, display_name, category, ...]`  
   - **`project_to_repositories.csv`**: Mapping from project_id to repository_id.  
   - **`developers_to_repositories.csv`**: Links developers to repositories with an `event_type`, `first_event`, `total_events`, etc.  
   - **`package_links.csv`**: Onchain -> Devtool package dependencies with columns such as `onchain_builder_project_id, devtooling_project_id, dependency_source`, etc.

3. **Outputs**:
   - A **CSV** file listing devtool projects (with their `pagerank_score`, etc.).  
   - A **JSON** file containing both onchain and devtool projects with additional metrics (number of commits, devs who contributed, etc.).

---

## Algorithm Steps

1. **Onchain Importance**  
   For each onchain project $O$, we compute  
   $\text{onchain\_importance}(O) = \alpha \tilde{T}_O + \beta \tilde{U}_O + \gamma \tilde{G}_O$
   where $\tilde{T}, \tilde{U}, \tilde{G}$ are **min-max normalized** values of transactions, active users, and gas fees, respectively. The final **onchain_importance** is used for personalization in PageRank.

2. **Package Dependency Edges** (Onchain -> Devtool):  
   - Base weight = `base_dependency_weight` (from YAML)  
   - Additional **source factor** = `dependency_source_weights[source]`  
   - e.g., if a project uses a Cargo package from a devtool, multiply the edge weight by `2.0` if that's the chosen factor in `dependency_source_weights`.

3. **Developer Connections**:
   1. **Developer Trust**: We measure how active or credible a developer is by factoring in how many onchain projects they contributed to, how many events they triggered, and the fraction that were commits.  
   2. **Event Weight**: Each event type (e.g. `COMMIT_CODE`, `FORKED`, `STARRED`) has a base weight from the YAML.  
   3. **Time Decay**: Event weight is halved every `half_life_days` (180 by default).  
   4. **Fanout Exponent**: If a developer engages with many devtools, the link to each devtool is penalized by `1 / (fanout_size^fanout_exponent)`.  
   5. Summation across all relevant devs: For each (Onchain -> Devtool) pair, we add up these partial weights.

4. **Construct Weighted Graph**:
   - **Node**: Each onchain or devtool project.  
   - **Edge**: Weighted by package usage, developer trust, time decay, etc.  
   - No self-loops (onchain -> same onchain).

5. **Compute PageRank**:
   - Using the standard formula with damping factor $\lambda$ (0.85 by default):  
     $r(v) = (1-\lambda)\text{personalization}(v) + \lambda \sum_{u \in \text{In}(v)} \frac{w(u,v)}{\sum_{x \in \text{Out}(u)} w(u,x)} r(u)$
   - **Personalization**: Onchain nodes get a share in proportion to their `onchain_importance`. Devtool nodes have zero (or minimal) personalization.  

6. **Export**:
   - **CSV**: By default, the final devtool ranking sorted by `pagerank_score`.  
   - **JSON**: Contains both onchain (with total commits, etc.) and devtool (with event-based metrics, references from onchain, etc.) data.

---

## Tuning Parameters

Below are key parameters you can fine-tune in the YAML:

1. Onchain Importance:
   - alpha, beta, gamma let you adjust how heavily each onchain metric (transactions, users, gas) influences the personalization vector.
2. Package Dependency:
   - base_dependency_weight: Overall strength of direct package edges.
   - dependency_source_weights: Additional multipliers by source (Cargo=2, NPM=1, etc.).
3. Developer Connections:
   - Event weights: e.g., COMMIT_CODE: 3 means commits are thrice as "valuable" as a default event.
   - Fanout exponent: The higher it is, the more you penalize devs who engage with many devtools.
   - Time decay: half_life_days controls how quickly old events lose weight.
4. PageRank:
   - damping_factor: Usually 0.85, but can range 0.7–0.95.
   - max_iterations, convergence_threshold: For controlling iteration limits and numerical convergence.

By adjusting these parameters, you can shape the final ranking to reflect the real importance of devtool usage, developer activity, and onchain project significance in your ecosystem.