import requests
import pandas as pd
from google.cloud import bigquery

# 1. EXTRACT


def extract_data():

    print("Extracting data...")

    url = "https://jsonplaceholder.typicode.com/users"

    response = requests.get(url)

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data)

    return df


# 2. TRANSFORM


def transform_data(df):

    print("Transforming data...")

    df = df.drop_duplicates(subset=["id"])

    df["city"] = df["address"].apply(lambda x: x["city"])

    df["company_name"] = df["company"].apply(lambda x: x["name"])

    df["name"] = df["name"].str.title()

    df["email"] = df["email"].str.lower()

    df = df[["id", "name", "username", "email", "phone", "city", "company_name"]]

    return df


# 3. LOAD


def load_data(df):

    print("Loading data to BigQuery...")

    client = bigquery.Client(project="netsol-etl-project")

    table_id = "netsol-etl-project.etl_data.customers"

    job = client.load_table_from_dataframe(df, table_id)

    job.result()

    print("Data loaded to BigQuery successfully!")


# MAIN PIPELINE


def run_etl():

    data = extract_data()

    transformed_data = transform_data(data)

    print(transformed_data)

    load_data(transformed_data)


run_etl()
