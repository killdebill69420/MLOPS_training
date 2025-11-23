import batch
import boto3
from datetime import datetime
import pandas as pd
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def prepare_dataframe():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    return pd.DataFrame(data, columns=columns)

def test_Q5():
    df_input = prepare_dataframe()
    bucket_name = "nyc-duration"
    file_name = "yellow_tripdata_2023-01.parquet"
    s3_endpoint = os.getenv("S3_ENDPOINT_URL")

    options = {"client_kwargs": {"endpoint_url": s3_endpoint}} if s3_endpoint else None
    input_file = f"s3://{bucket_name}/{file_name}"

    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def test_Q6():
    batch.main(2023, 1)


if __name__ == "__main__":
    test_Q5()
    test_Q6()