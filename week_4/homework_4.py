#!/usr/bin/env python
import pickle
import pandas as pd
import numpy as np
import os
import argparse

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    print("Creating dataframe...")
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def load_data(year, month):
    print("Loading data...")
    return read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet')

def train_model(df):
    print("Training model...")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(y_pred.mean())

    return y_pred


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str)
    parser.add_argument('--month', type=str)
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    year = args.year
    month = args.month

    df = load_data(year, month)
    y_pred = train_model(df)
    return y_pred.mean()

if __name__ == '__main__':
    main()

