# e2e-bike-count-predictor

A bike-sharing prediction service that can be useful to plan daily supply of bikes based on historical data

## Problem Description

### Overview

The problem I have chosen to work on is determining the age of abalones by predicting the number of rings on their shells. The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope – a boring and time-consuming task. Other measurements, which are easier to obtain, may be used to predict the age. The number of rings is the value to predict.
This will be treated as a regression problem.

### Variables

Variables from left to right on the `abalone.csv` file are:

* Sex: M, F, and I (infant)
* Length: Longest shell measurement (mm)
* Diameter: perpendicular to length (mm)
* Height: with meat in shell (mm)
* Whole weight: whole abalone (grams)
* Shucked weight: weight of meat (grams)
* Viscera weight: gut weight after bleeding (grams)
* Shell weight: after being dried (grams)
* Rings: +1.5 gives the age in years

These variables were measured on 4177 abalones.

### References

* Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994) ”The Population Biology of Abalone (Haliotis species) in Tasmania. I. Blacklip Abalone (H. rubra) from the North Coast and Islands of Bass Strait”, Sea Fisheries Division, Technical Report No. 48.

### Data Source

UCI machine learning Repository.\`

## Preparing the Dataset files

The data set used for this project was downloaded from [here](https://sci2s.ugr.es/keel/dataset.php?cod=96). It contains a total of 4177 samples and 8 features (1 target variable).
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
Experiment tracking is done using MLFlow. The best estimators i.e the model with the lowest `validation_rmse` for a hyperparameter configuration of the regression models is registered in MlFlow's model registry.
The chosen best model is then transitioned to "production" stage in the model registry.
The model training is orchestrated via a fully deployed workflow with Prefect.
This same best model, is merged with the preprocessing pipeline to make a single `.pkl` Python object and is logged to a new experiment (the new experiment name has "\_production" appended to previous experiment name). This combined preprocessor and model, as a single object, will be used for making predictions will in deployment.

### Model Deployment

Model is deployed using batch deployment method.
The orchestration of this is fully automated by Prefect, with the prediction and model monitoring scheduled to happen on the second day of every month. Input data, as well as reference data required for measuring data drift is loaded from an s3 bucket storage. The combined, single object preprocessor and model, is loaded and applied to both input and reference to make predictions.
These prediction results are then fed to Evidently which uses it to calculate data drift and the regression model performance. The results of this calculations are then used to generate interactive html reports, which contain visuals, charts and metrics which could be further explored.