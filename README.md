# Help summon Ether's Phoenix
[Ether’s Phoenix](https://medium.com/ethereum-optimism/ethers-phoenix-18fb7d7304bb) is a magical creature that rewards those who help build a world where impact = profit.
It is an evolving algorithm that ensures public goods and positive contributions are recognized and fairly rewarded. You can help summon and grow Ether’s Phoenix by contributing **data**, **models**, **metrics**, and **feedback** to improve how Retroactive Public Goods Funding measures impact. 

Together, we can rewrite the rules, ensuring that those who create impact are fairly rewarded for it.

**Current State - Evolution 1: baby phoenix**

<p align="center">
  <img src="https://github.com/user-attachments/assets/d1b4394c-fc3c-40d2-af1f-a0ed3fd1ae2a" alt="EggAnimated">
</p>



# Quickstart

## Setup

Clone the repo:

```bash
gh repo clone ethereum-optimism/Retro-Funding
```

Navigate to the repo:

```bash
cd Retro-Funding
```

Install the dependencies:

```bash
poetry install
```

Activate the virtual environment:

```bash
poetry shell
```

Then, you can execute the commands below to test the simulation pipeline.

## Simulation Pipeline

> [!NOTE]
> We have provided raw data files in the `data` directory for testing purposes. These are NOT the latest data and do not reflect the actual projects applying for Retro Funding. This is only for testing purposes.

Modify the YAML config files in `eval-algos/S7/weights` to update the metrics, weights, and other parameters.

Then, from the root directory, run:

```bash
python eval-algos/S7/models/onchain_builders.py
python eval-algos/S7/models/devtooling_openrank.py
```

This will run the allocation pipeline and save the results to the corresponding data directories using the default"testing" weights.

To run the pipeline with a specific set of weights, run the scripts with the desired weights file as an argument:

```bash
python eval-algos/S7/models/onchain_builders.py onchain_builders_goldilocks.yaml
python eval-algos/S7/models/devtooling_openrank.py devtooling_arcturus.yaml
```

You can now make your own changes to the weights and run the pipeline again, eg:

```bash
python eval-algos/S7/models/onchain_builders.py <my_weights.yaml>
```

Here is a [Loom Video tutorial](https://www.loom.com/share/75484a94fe404b0a9d9b09c82938d0cb?sid=45ffdb03-e9ac-4b04-8bd0-7d556171d661) for further guidance.

## Simulate Funding Allocation

We have also provided a module to allocate funding to the projects based on the results from the simulation pipeline.

You can see how this works by looking at the [TestAlgo notebook](./TestAlgo.ipynb).

For example:

```python
import pandas as pd
import sys

sys.path.append('./eval-algos/S7/models/')
from onchain_builders import OnchainBuildersCalculator, load_config, load_data
from utils.allocator import AllocationConfig, allocate_with_constraints

# Load the data and config
ds, sim_cfg = load_config('eval-algos/S7/weights/onchain_builders_testing.yaml')
calculator = OnchainBuildersCalculator(sim_cfg)
df_data = load_data(ds)

# Run the analysis
analysis = calculator.run_analysis(df_data)

# Get the scores (and use the display_name as the index)
scores = analysis['final_results']['weighted_score'].reset_index().set_index('display_name')['weighted_score']

# Configure the budget and allocation constraints
alloc = AllocationConfig(
  budget=1_000_000,
  min_amount_per_project=200,
  max_share_per_project=0.05
)

# Allocate the funding
rewards = allocate_with_constraints(scores, alloc)
```

# Model Details

## Onchain Builders

The Onchain Builders model analyzes and scores active Superchain projects based on their onchain activity. It processes raw metrics through a multi-step pipeline to produce normalized, weighted scores that reflect project performance across different chains and time periods.

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

# Data Updates

## Credentials

Add your BigQuery credentials to the root directory as `credentials.json`. Update the `GCP_CREDENTIALS` and `GCP_PROJECT_ID` in `eval-algos/scripts/config.py` to match your project.

See [here](https://docs.opensource.observer/docs/get-started/bigquery) for more information on how to get your BigQuery credentials.

## Retrieve the Latest Data

From the root directory, run:

```bash
python eval-algos/scripts/fetch_data.py
```

This will download the data from BigQuery and save it to the `data` directory. You can review the included models in `eval-algos/scripts/config.py`.

Note: the models in the config file may not always be the same as the latest ones used for the algorithms, so for the time being, we recommend using the raw data files in the `data` directory.
