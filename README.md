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

Note: we have provided raw data files in the `data` directory for testing purposes.

# Model Details

## Onchain Builders

The Onchain Builders model analyzes and scores blockchain-based projects based on their onchain activity. It processes raw metrics through a multi-step pipeline to produce normalized, weighted scores that reflect project performance across different chains and time periods.

<details>
<summary>Pipeline Steps</summary>

#### 1. Filter and Pivot Raw Metrics
- Takes raw metrics data with non-zero weights for specified measurement periods
- Pivots data by chain and metric to create a structured view
- Groups by `project_id`, `project_name`, `display_name`, and `chain`

#### 2. Sum and Weight by Chain
- Applies chain-specific weights (e.g., different weights for OP Mainnet vs other chains)
- Sums weighted metrics across all chains for each project
- Preserves project metadata in the aggregation

#### 3. Calculate Metric Variants
For each metric, computes three variants:
- **Adoption**: Current period value
- **Growth**: Positive difference between current and previous period values
- **Retention**: Minimum value between current and previous periods

#### 4. Normalize Metric Variants
- Applies min-max normalization to each metric variant
- Scales values to [0,1] range while preserving null values (e.g., TVL for non-DeFi projects)
- Uses fallback center value (0.5) when range is zero

#### 5. Apply Weights
- Multiplies each normalized metric variant by:
  - Its metric-specific weight
  - Its variant-specific weight (adoption/growth/retention)

#### 6. Aggregate Final Scores
- Combines weighted variants using power mean (p=2)
- Normalizes final scores to sum to 1.0

#### 7. Prepare Results
- Flattens multi-level columns for readability
- Merges intermediate results for transparency
- Sorts projects by final weighted score
</details>

## Devtooling OpenRank

The Devtooling model evaluates open-source developer tools by analyzing their relationships with onchain projects through package dependencies and developer contributions. It uses an EigenTrust-based algorithm released by OpenRank to propagate value through the graph.

<details>
<summary>Pipeline Steps</summary>

#### 1. Build Unweighted Graph
Constructs a directed graph with three types of edges:
- **Package Dependencies**: Onchain projects → Devtooling projects
- **Commit Events**: Onchain projects → Developers
- **GitHub Engagement**: Developers → Devtooling projects

Removes duplicate edges when an onchain project is also a devtooling project.

#### 2. Compute Onchain Project Pretrust
- Uses economic metrics from onchain projects
- Applies log transformation and min-max scaling
- Combines metrics using configured weights
- Normalizes scores to sum to 1.0

#### 3. Compute Devtooling Project Pretrust
- Uses GitHub metrics (num packages, stars, forks, etc.)
- Applies log transformation and min-max scaling
- Combines metrics using configured weights
- Normalizes scores to sum to 1.0

#### 4. Compute Developer Reputation
- Developers are pre-filtered to focus on active developers committing code to onchain project repos in Rust, Solidity, TypeScript, and Vyper
- Distributes onchain project pretrust to developers based on commit activity

#### 5. Weight Edges
Applies weights based on:
- Link type (package dependency, commit, GitHub engagement)
- Event type (NPM, CARGO, COMMIT_CODE)
- Time decay for non-static relationships
- More recent interactions contribute more weight in most cases

#### 6. Apply EigenTrust
- Combines pretrust scores from all sources
- Runs EigenTrust propagation on weighted graph
- Computes final trust scores for each node

#### 7. Rank and Evaluate Projects
- Ranks devtooling projects by final EigenTrust scores
- Applies eligibility criteria:
  - Minimum package dependency count
  - Minimum developer link count
- Normalizes scores among eligible projects

#### 8. Serialize Value Flow
- Uses iterative proportional fitting (IPF)
- Creates detailed value flow attribution
- Ensures contribution sums match:
  - Per devtool: Sum equals its overall score
  - Per onchain project: Sum equals its pretrust
</details>