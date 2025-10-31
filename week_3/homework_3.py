import pandas as pd
import os
import yaml
import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

categorical = ['PULocationID', 'DOLocationID']
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Homework 3")
mlflow.sklearn.autolog()

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df[categorical] = df[categorical].astype(str)

    return df

def create_X(df, dv=None):
    dicts = df[categorical].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train, df_train):
    with mlflow.start_run() as run:
        target = 'duration'
        y_train = df_train[target].values
        model = LinearRegression()

        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, artifact_path="model")

        y_pred = model.predict(X_train)
        rmse = sqrt(mean_squared_error(y_train, y_pred))
        mlflow.log_metric("rmse", rmse)
        print(model.intercept_)

        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="Homework 3")

    return model_uri

def find_model(model_uri):
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    mlmodel_path = os.path.join(local_path, "MLmodel")

    with open(mlmodel_path, "r") as f:
        mlmodel_data = yaml.safe_load(f)

    model_size = mlmodel_data.get("model_size_bytes", None)
    print("Model size (bytes):", model_size)

def run():
    df = read_dataframe("./data/yellow_tripdata_2023-03.parquet")

    X_train, dv = create_X(df)

    model_uri = train_model(X_train, df)

    find_model(model_uri)

if __name__ == "__main__":
    run()
