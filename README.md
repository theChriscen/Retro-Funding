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

This will download the data from BigQuery and save it to the `data` directory. You can review the included models in `eval-algos/scripts/config.py`.
