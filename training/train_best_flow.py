import pickle
import pathlib

import mlflow
import pandas as pd
import xgboost as xgb
from prefect import flow, task
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


@task(retries=3, retry_delay_seconds=2)
def read_data():
    # pylint: disable=anomalous-backslash-in-string
    ref_data_path = '../data/interim/ref_data.csv'
    ref_data = pd.read_csv(ref_data_path)

    ref_data.columns = (
        ref_data.columns.str.lower()
        .str.replace("\s*\(.*\)\s*", "", regex=True)
        .str.replace(' ', '_')
    )
    return ref_data


@task
def features_engineering(data: pd.DataFrame):
    '''transform/select features'''
    # from date column (string format) create new column
    # with progressive day numbers (int type)
    data['day_number'] = pd.to_datetime(data['date'], dayfirst=True)
    data['day_number'] = (data['day_number'] - data['day_number'].min()).dt.days + 1
    data['day_number'] = data['day_number'].map(
        {value: index + 1 for index, value in enumerate(data['day_number'].unique())}
    )

    # add a column with day of the week
    data['weekday'] = pd.to_datetime(data['date'], dayfirst=True).dt.strftime('%A')

    # filtering out rows for not functioning days
    # (deterministic relation: no_functioning -> no rented bike for that day)
    data = data.loc[data['functioning_day'] == 'Yes']

    # qualitative maanual feature selection from EDA
    data = data[
        [
            'temperature',
            'humidity',
            'hour',
            'day_number',
            'rainfall',
            'seasons',
            'weekday',
            'rented_bike_count',
        ]
    ]

    return data


@task
def prepare_dictionaries(df: pd.DataFrame):
    '''prepare data for model application'''
    numerical = list(df.select_dtypes(include=['int64', 'float64']).columns)
    categorical = list(df.select_dtypes(include='object').columns)

    dicts = df[categorical + numerical].to_dict(orient='records')

    return dicts


@task(log_prints=True)
def train_best_model(X_train, X_val, y_train, y_val, dv):
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.1615338857877632,
            "max_depth": 46,
            "min_child_weight": 11.5291496309669,
            "objective": "reg:linear",
            "reg_alpha": 0.027104850675872743,
            "reg_lambda": 0.10235774157109881,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=10,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models")


@flow
def main_flow():
    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bike-count-experiment")

    # Load
    ref_data = read_data()

    # Transform
    data_prep = features_engineering(ref_data)

    y = data_prep.rented_bike_count.values
    X = data_prep.drop(columns='rented_bike_count', axis=1)

    df_train, df_val, y_train, y_val = train_test_split(
        X, y, random_state=42, stratify=X['seasons']
    )

    dv = DictVectorizer()

    train_dicts = prepare_dictionaries(df_train)
    X_train = dv.fit_transform(train_dicts)

    val_dicts = prepare_dictionaries(df_val)
    X_val = dv.transform(val_dicts)

    train_best_model(X_train, X_val, y_train, y_val, dv)


if __name__ == '__main__':
    main_flow()
