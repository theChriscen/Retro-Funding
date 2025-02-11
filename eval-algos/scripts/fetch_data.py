import os
from google.cloud import bigquery
from datetime import datetime
import pandas as pd

from config import (
    GCP_CREDENTIALS,
    GCP_PROJECT_ID,
    MODELS,
    EXPORT_DIR
)


def setup_bigquery_client():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_CREDENTIALS
    client = bigquery.Client(project=GCP_PROJECT_ID)
    return client


def ensure_directories(archive_dir=EXPORT_DIR):
    os.makedirs(archive_dir, exist_ok=True)
    return archive_dir


def fetch_and_save_data(client, archive_dir, model_name):
    current_date = datetime.now()
    date_prefix = current_date.strftime('%Y%m%d')
    
    query = f"""select * from `{GCP_PROJECT_ID}.oso_production.{model_name}`"""
    df = client.query(query).to_dataframe()

    model_short_name = model_name.replace('int_superchain_s7_', '')
    filename = f"{date_prefix}_{model_short_name}.csv"
    output_path = os.path.join(archive_dir, filename)
    df.to_csv(output_path, index=False)
    return filename

def main(output_dir):
    client = setup_bigquery_client()
    archive_dir = ensure_directories(output_dir)
    
    for model_name in MODELS:
        print(f"Fetching {model_name}")
        try:
            filename = fetch_and_save_data(client, archive_dir, model_name)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")

if __name__ == "__main__":
    output_directory = EXPORT_DIR
    main(output_directory)
