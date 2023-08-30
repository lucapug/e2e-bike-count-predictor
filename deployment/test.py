import requests

hour_record = {
    "Date": ["01/12/2017"],
    "Rented Bike Count": [254],
    "Hour": [0],
    "Temperature(°C)": [-5.2],
    "Humidity(%)": [37],
    "Wind speed (m/s)": [2.2],
    "Visibility (10m)": [2000],
    "Dew point temperature(°C)": [-17.6],
    "Solar Radiation (MJ/m2)": [0.0],
    "Rainfall(mm)": [0.0],
    "Snowfall (cm)": [0.0],
    "Seasons": ["Winter"],
    "Holiday": ["No Holiday"],
    "Functioning Day": ["Yes"],
}


url = 'http://localhost:9696/predict'
response = requests.post(url, json=hour_record, timeout=5)
print(response.json())
