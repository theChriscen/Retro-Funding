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
python eval-algos/models/onchain_builders.py
# python eval-algos/models/dev_tooling_pagerank.py
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

The Devtooling model focuses on open-source development projects, especially developer tools, by looking at how developers interact with them, how tools depend on each other, and how these projects connect to onchain ecosystems. It uses a specialized PageRank approach to highlight which devtools are central and actively maintained.

<details>
<summary>Steps and Logic</summary>

#### 1. Collect Devtool Data
Obtain information on repositories, developer contributions (e.g., commits and forks), and links between devtool projects and onchain builder projects.

#### 2. Onchain Importance
Borrow a concept of "onchain importance" from the onchain builders model. Projects with more transactions or users can give higher "trust" to developers who also contribute to devtool projects.

#### 3. Developer Mapping
Identify which developers work on onchain projects, devtool projects, or both. This step helps link devtool projects to their contributors' experience.

#### 4. Shared Developer Activity
If the same developers contribute to an onchain project and a devtool, the devtool is considered more connected and potentially more valuable.

#### 5. Package Dependencies
A devtool that multiple onchain projects or other devtools depend on may be more influential. These dependencies create directed edges that feed into the PageRank calculation.

#### 6. PageRank Graph
Build a graph of nodes (onchain and devtool projects), with edges capturing developer overlap, time decay of contributions, and package dependencies. Run a PageRank-like algorithm that uses onchain importance to personalize the ranking of devtool projects.

#### 7. Devtool Metrics
In addition to PageRank scores, gather metrics like the number of commits from onchain-experienced developers, total dev contributors, and how many onchain projects reference a particular devtool.

#### 8. Results
The final output ranks devtool projects by their overall PageRank score, along with extra metrics for context. This reveals which devtools have high developer engagement, strong connections to key onchain projects, and overall strategic importance.
</details>