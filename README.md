# e2e-bike-count-predictor

A bike-sharing prediction service is a useful tool to plan daily supply of bikes. The main goal is to create an end to end service with the main characteristics required in a production environment: possibility of monitoring in production with possible consequent refinement of the prediction model, reproducibility of the entire workflow, quality checks and automation.

## Problem Description

### Overview

The problem is assisting a bike-sharing activity in taking decisions about the supply of bikes over time, to promptly react to demand fluctuations. The specific objective is to predict the hourly count of rented bikes. For this project, the predictions are based on historical data from the bike-sharing system of the city of Seoul, in South Korea. Ddareungi (Seoul Bike) is Seoul, South Korea’s bike-sharing system. Ddareungi started as a leisure activity but has transformed into a popular means of transportation. This will be treated as a regression problem.

### Data Source

There is a dataset originally extracted from the official Seoul Open Data Plaza and that was already used in scientific research activities (see references). It is publicly available in the **UCI machine learning Repository** and it can be downloaded from [here](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand). 

### Dataset description

It is a tabular dataset, comprising **8760 rows** and <strong>14 columns</strong>. Here below the name and short explanation of each column (variable):

* Date : year-month-day 
* Rented Bike count : Count of bikes rented at each hour 
* Hour : Hour of he day 
* Temperature : Temperature in Celsius 
* Humidity : % 
* Windspeed : m/s 
* Visibility : 10m 
* Dew point temperature : Temperature in Celsius 
* Solar radiation : MJ/m2 
* Rainfall : mm 
* Snowfall : cm 
* Seasons : Winter, Spring, Summer, Autumn 
* Holiday : Holiday/No holiday 
* Functional Day : NoFunc(Non Functional Hours), Fun(Functional hours)

The data stored in a row are values for a fixed hour of the day. The data spans from December, 1, 2017 to November, 30, 2018. The **Rented Bike count** is the target variable in this problem. 

### References

* Sathishkumar V E & Yongyun Cho (2020). A rule-based model for Seoul Bike sharing demand prediction using weather data, <em>European Journal of Remote Sensing</em>, DOI:10.1080/22797254.2020.1725789
* Ngo, T.T., Pham, H.T., Acosta, J.G., & Derrible, S. (2022). Predicting Bike-Sharing Demand Using Random Forest. <em>Journal of Science and Transport Technology</em>.

### 

## Preparing the Dataset files


From the total samples, 600 were set aside. These 600 samples were saved to external file, `reserved_data.csv`.
To simulate monthly data when deploying in batch mode, the 600 samples were divided into 12 batches (each batch containing 50 samples), and each batch was saved to a csv file (`reserved_1.csv`, `reserved_2.csv`, ..., `reserved_12.csv`).
These files are uploaded to s3 bucket at this location: "s3://mlops-zoomcamp-datasets/live/".
The remaining 3577 samples are saved into a separate csv file, `abalone_data.csv`. The data in this file will be used to train the model; this will also serve as the reference data for calculating data drift, and regression model performance during model monitoring.
File is uploaded to s3 bucket at this location: "s3://mlops-zoomcamp-datasets/reference-data/abalone\_data.csv". This is the location the Prefect Deployment of the model training gets the training data from. This is also the location the monitoring service gets the reference data from.
The file containing the code that does this splitting is `prepare_data.ipynb`.

### Training

* Scaling of numerical features is done with `MinMaxScaler`
* Scaling of categorical features is done with `OneHotEncoder`

The model is trained by running a randomized search over the following regression models `RandomForestRegressor`, `ElasticNet`, `Ridge`, `LinearRegression` and `Lasso` across a range of hyperparameters.

**Experiment tracking** is done using MLFlow. The best estimators i.e the model with the lowest `validation_rmse` for a hyperparameter configuration of the regression models is registered in MlFlow's model registry.
The chosen best model is then transitioned to "production" stage in the model registry.

**The model training is orchestrated** via a fully deployed workflow with Prefect.
This same best model, is merged with the preprocessing pipeline to make a single `.pkl` Python object and is logged to a new experiment (the new experiment name has "\_production" appended to previous experiment name). This combined preprocessor and model, as a single object, will be used for making predictions will in deployment.

### Model Deployment

Model is deployed using batch deployment method.
The orchestration of this is fully automated by Prefect, with the prediction and model monitoring scheduled to happen on the second day of every month. Input data, as well as reference data required for measuring data drift is loaded from an s3 bucket storage. The combined, single object preprocessor and model, is loaded and applied to both input and reference to make predictions.

**These prediction results are then fed to Evidently** which uses it to calculate data drift and the regression model performance. The results of this calculations are then used to generate interactive html reports, which contain visuals, charts and metrics which could be further explored.