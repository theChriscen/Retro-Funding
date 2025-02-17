## Guide for Retrieving Data Snapshots from OSO

### Setup

Clone the repo and install the dependencies:

```bash
poetry install
```

Add your BigQuery credentials to the root directory as `credentials.json`. Update the `GCP_CREDENTIALS` and `GCP_PROJECT_ID` in `eval-algos/scripts/config.py` to match your project.

See [here](https://docs.opensource.observer/docs/get-started/bigquery) for more information on how to get your BigQuery credentials.

### Retrieve the Latest Data

From the root directory, run:

```bash
python eval-algos/scripts/fetch_data.py
```

This will download the data from BigQuery and save it to the `data` directory. You can review the included models in `eval-algos/models/config.py`.


### Run the Allocation Pipeline

Modify the YAML config files in `eval-algos/S7/weights` to update the metrics, weights, and other parameters.

Then, from the root directory, run:

```bash
python eval-algos/S7/models/onchain_builders.py
python eval-algos/S7/models/devtooling_openrank.py
```

This will run the allocation pipeline and save the results to the corresponding data directories.

# Model Details (WIP)

## Onchain Builders

The Onchain Builders model is designed to analyze and score blockchain-based projects based on their onchain activity. By focusing on relevant metrics (like transactions, active users, and gas usage) and comparing them across different time periods, this model can help identify projects that are growing, retaining users, or adopting new behavior at a higher rate.

<details>
<summary>Steps and Logic</summary>

#### 1. Collect Onchain Data
Gather metrics—such as transaction counts, user counts, and possibly gas fees—for each project from the relevant blockchains. Each metric is associated with a specific time period (for instance, a "current" month versus a "previous" month).

#### 2. Pivoting and Aggregation
Reorganize the raw data so that projects become rows, and each relevant (time period, metric) pair becomes a column. This allows for side-by-side comparisons of metrics for each project.

#### 3. Chain Weighting
Some blockchains may carry more significance than others. For example, you can assign a higher weight to a project's performance on one chain over another. The model multiplies each project's metrics by the chain's weight before summing.

#### 4. Metric Variants
Instead of only using raw metrics, the model breaks them down into:
- Adoption: The current level of usage or activity
- Growth: The change compared to the previous period
- Retention: A measure of continuous usage, such as the smaller value between the current and previous metrics

#### 5. Normalization
Different metrics may have vastly different scales (e.g., user counts vs. gas fees). The model normalizes each metric to make them comparable, often on a zero-to-one scale.

#### 6. Weighting
Specify how important each metric is. For example, you might value transaction counts more than user counts. These weights are applied to the normalized metrics, reflecting your priorities.

#### 7. Final Score
After combining the weighted metrics, each project receives a single aggregated score, which can be sorted from highest to lowest. This provides a clear ranking of projects based on their onchain performance.

#### 8. Results
The final output is a table showing each project's contributions to the overall score and the final ranking. Optionally, you can also apply an allocation mechanism to distribute budgets or rewards proportionally based on those scores.
</details>

## Devtooling

The Devtooling model evaluates the value of open-source developer tool projects by quantifying their usefulness to onchain projects. It integrates economic signals from onchain projects, developer contributions, and GitHub activity into a unified trust propagation framework using an EigenTrust algorithm. This model highlights which devtool projects are most central and impactful by considering both direct dependencies and the influence of developer activity.

<details>
<summary>Steps and Logic</summary>

#### 1. Data Collection
- **Gather Data:**  
  Collect data on:
  - **Onchain Projects:** Metrics such as transaction counts, gas fees, and user activity.
  - **Devtooling Projects:** GitHub metrics such as star counts, forks, etc.
  - **Package Dependencies:** Relationships where onchain projects depend on devtooling projects.
  - **Developer Contributions:** Commit events and other GitHub engagement data linking developers to projects.

#### 2. Pretrust Seeding
- **Onchain Project Pretrust:**  
  Compute economic pretrust scores for onchain projects by applying a log transformation and min–max scaling to metrics (e.g., transaction counts, gas fees). These scores are combined using configured weights and normalized.
- **Devtooling Project Pretrust:**  
  Similarly, compute GitHub-based pretrust scores for devtooling projects using metrics like star count and fork count.
- **Developer Reputation:**  
  Distribute onchain project pretrust scores to developers based on their commit activity. The resulting developer reputation is normalized and reflects how much trust developers have earned through their onchain contributions.

#### 3. Graph Construction
- **Directed Edges:**  
  Build an unweighted directed graph with three types of edges:
  1. **Package Dependency:** Onchain projects → Devtooling projects.
  2. **Commit Events:** Onchain projects → Developers.
  3. **GitHub Engagement:** Developers → Devtooling projects.
- **Duplicate Removal:**  
  Remove duplicate edges when an onchain project also appears as a devtooling project, ensuring no overcounting of trust contributions.

#### 4. Edge Weighting & Time Decay
- **Weight Assignment:**  
  Each edge is weighted based on:
  - Its link type (e.g., package dependency, commit event, GitHub engagement).
  - Its event type (e.g., NPM, CARGO, COMMIT_CODE) via configured weights.
- **Time Decay:**  
  Apply an exponential decay factor to edges (except static package dependencies) based on the recency of the event. More recent interactions contribute more to the final score.

#### 5. Trust Propagation (EigenTrust)
- **Combined Pretrust:**  
  Merge pretrust scores from onchain projects, devtooling projects, and developer reputation to create a unified seed.
- **EigenTrust Algorithm:**  
  Run the EigenTrust propagation on the weighted graph to compute final trust (or OpenRank) scores for each node, capturing the overall influence of each project.

#### 6. Ranking & Eligibility
- **Devtooling Ranking:**  
  Devtooling projects are ranked based on their final EigenTrust scores.
- **Eligibility Criteria:**  
  Projects must meet configured thresholds (e.g., minimum counts of onchain package dependencies or developer links) to be considered eligible.
- **Normalization:**  
  The final scores are normalized so that the sum of scores among eligible projects equals 1.

#### 7. Value Flow Graph
- **Detailed Attribution:**  
  Use an iterative proportional fitting (IPF) procedure to create a detailed value flow graph:
  - For each devtooling project, the sum of contributions from onchain projects equals its overall score.
  - For each onchain project, the total contributions equal its economic pretrust.
- **Visualization:**  
  The resulting data can be used to generate Sankey diagrams or other visualizations that show how trust flows from onchain projects to devtooling projects.

#### 8. Results
- **Final Outputs:**  
  - A ranked list of devtooling projects with normalized trust scores.
  - Detailed relationship data capturing the contribution of each onchain project to each devtooling project’s score.
  - Supplementary metrics (e.g., counts of developer links and package dependencies) providing additional context.
  
</details>