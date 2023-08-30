import pickle

import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request

with open('preprocessor.b', 'rb') as f_in:
    dv = pickle.load(f_in)

with open('best_model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

# pylint:disable=pointless-string-statement
""" model = mlflow.xgboost.load_model(
    '../training/mlruns/1/e46e319eb44d4fddbb73f54a1d1c36dc/artifacts/models/'
)

with open('best_model.bin', 'wb') as f_out:
    pickle.dump(model, f_out) """


def features_engineering(data: pd.DataFrame):
    # pylint:disable=anomalous-backslash-in-string
    """transform/select features

    :param data: raw data
    :type data: pd.DataFrame
    :return: data ready for stat analysis
    :rtype: pd.DataFrame
    """
    data.columns = (
        data.columns.str.lower()
        .str.replace("\s*\(.*\)\s*", "", regex=True)
        .str.replace(' ', '_')
    )

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


def predict(features, y):
    X = dv.transform(features)
    X1 = xgb.DMatrix(X, label=y)
    preds = model.predict(X1)
    return float(preds[0])


app = Flask('bike-count-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    hour_record_json = request.get_json()
    hour_record = pd.DataFrame.from_dict(hour_record_json)

    data_prep = features_engineering(hour_record)

    y = data_prep.rented_bike_count.values
    features = data_prep.drop(columns='rented_bike_count', axis=1)

    prep_dicts = prepare_dictionaries(features)

    pred = predict(prep_dicts, y)

    result = {'bike-count': pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
