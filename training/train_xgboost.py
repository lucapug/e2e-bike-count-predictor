import mlflow
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bike-count-experiment")


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


def prepare_dictionaries(df: pd.DataFrame):
    '''prepare data for model application'''
    numerical = list(df.select_dtypes(include=['int64', 'float64']).columns)
    categorical = list(df.select_dtypes(include='object').columns)

    dicts = df[categorical + numerical].to_dict(orient='records')

    return dicts


def train_xgboost(X_train, X_val, y_train, y_val):
    # pylint: disable=unused-variable
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=10,
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42,
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
    )


def main():
    ref_data = read_data()
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

    train_xgboost(X_train, X_val, y_train, y_val)


if __name__ == '__main__':
    main()
