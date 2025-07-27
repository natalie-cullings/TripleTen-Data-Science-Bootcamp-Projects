# Project 13: Time Series
#### *Taxi Order Demand Prediction Model*

## Table of Contents

1. [Overview & Objective](#overview_objective)
2. [Dataset](#dataset)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Overview & Objective <a id="overview_objective"></a>

The hypothetical taxi company Sweet Lift Taxi wants to attract more drivers to work during peak hours at airports. They have collected historical data on the hourly number of taxi orders at airports. 

The objective of this project is to build a time series model that will predict the number of taxi orders for the next hour. The RMSE metric for the test set can't exceed 48.


## Dataset <a id="dataset"></a>

This project relies on the provided `taxi.csv` data file, which contains the following two fields:
- ** datetime:** The day and hour in which the taxi rides occured.
- ** num_orders:** The total number of taxi rides ordered within the given time period.

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and Inspection:**
   - Import the dataset and inspect it.

2. **Data Resampling:**
    - Resample the data on an hourly level.

3. **Data Analysis**
    - Examine the trend, seasonality, and residuals of the data.
    - Determine if the data is stationary or not.

4. **Feature Engineering:**
    - Create additional features for the data.

5. **Split the Data into Training, Validation, and Testing Sets:**
    - Split the data into training, validation, and testing sets.

6. **Model Training:**
    - Calculate the RSME of various models including the linear regression, decision tree regressor, random forest regressor, autoregressive, moving average, ARIMA, and prophet models.

7. **Model Selection and Evaluation:**
   - Determine the best model and get the RSME of the test set.

8. **Insights**:
   - Draw conclusions about the project and the best model chosen.

## Requirements <a id="requirements"></a>

To run this project, the necessary Python dependencies can be installed with `pip`. Please follow these steps:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
   The `requirements.txt` file includes the following packages:
    - `matplotlib==3.8.4`
    - `numpy==2.1.3`
    - `pandas==2.2.3`
    - `scikit_learn==1.5.2`
    - `statsmodels==0.14.0`
    - `xgboost==2.1.2`
   

## Results <a id="results"></a>

The Random Forest Regressor model had the best RSME on the test set at a score of 42.1. With this accuracy, Sweet Lift Taxi can be confident in its ability to predict how many taxis will be orderd in the next hour. 

More specifically, the Random Forest Regressor model predicts that ~186 taxis will be ordered in the next hour (the hour that would come after the hour that the original dataset ended with). This information will help them to alert drivers about periods of high airport ride demand so they can entice more drivers to get on the road.
