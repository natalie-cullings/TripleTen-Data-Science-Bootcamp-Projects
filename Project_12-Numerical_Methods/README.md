# Project 12: Numerical Methods
#### *Market Value of a Car Prediction Model*

## Table of Contents

1. [Overview & Objective](#overview_objective)
2. [Dataset](#dataset)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Overview & Objective <a id="overview_objective"></a>

Rusty Bargain, a hypothetical used car sales service company is developing an app to attract new customers. The app allows users to quickly find out the market value of their car. 

Using historical data including technical specifications, trim versions, and prices, this project's objective is to build a model that utilizes gradient boosting; Rusty Bargain's new app will leverage this model to determine a vehicle's market value.


## Dataset <a id="dataset"></a>

This project relies on the provided `car_data.csv` data file. 

The dataset includes the following columns:
**Features**
- DateCrawled — The date that the profile was downloaded from the database.
- VehicleType — The vehicle body type.
- RegistrationYear — The vehicle registration year.
- Gearbox — The gearbox type.
- Power — The power (hp).
- Model — The vehicle model.
- Mileage — The mileage (measured in km due to the dataset's regional specifics).
- RegistrationMonth — The vehicle registration month.
- FuelType — The fuel type.
- Brand — The vehicle brand.
- NotRepaired — Whether the vehicle was repaired or not.
- DateCreated — The date of the profile creation.
- NumberOfPictures — The number of pictures of the vehicle.
- PostalCode — The postal code of profile owner (user).
- LastSeen — The date of the last activity of the user.
**Target**
- Price — The price (in Euros).

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and EDA:**
   - Import the datasets and perform an exploratory data analysis.

2. **Data Preprocessing:**
    - Drop unnecessary columns, clean the dataset, and impute missing values.

3. **Split the Data into Train, Validate, and Test Sets:**
    - Split the data into training, validation, and testing sets then specify features and a target for each.

4. **Model Training & Optimization:**
    - Train different models including Linear Regression, Random Forest Regressor, CatBoost Regressor, XGBoost, and LightGBM models.
    - Optimize each model's hyperparameters and select the best model.

5. **Calculate the Testing Set Score:**
   - Evaluate the best model on the test set.

6. **Insights**:
   - Draw conclusions about the project and the best model chosen.

## Requirements <a id="requirements"></a>

To run this project, the necessary Python dependencies can be installed with `pip`. Please follow these steps:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
   The `requirements.txt` file includes the following packages:
    - `catboost==1.2.7`
    - `lightgbm==4.5.0`
    - `matplotlib==3.8.4`
    - `numpy==2.1.3`
    - `optuna==4.0.0`
    - `pandas==2.2.3`
    - `scikit_learn==1.5.2`
    - `xgboost==2.1.2`
   

## Results <a id="results"></a>

The testing set performed slightly better than the validation set with an RMSE of 1,709. This means that the LightGBM model is, on average, able to predict the value of a vehicle with accuracy within a range of $1,709.46 USD!

The linear regression model was expected to perform the worst for Rusty Bargain's need and hence serves as a solid sanity check against the performance of the other models. The linear regression model achieved a RMSE of 2,980 so the other models are certainly an improvement. For example, the Random Forest Regressor model achieved a RMSE of 2,039, which was a solid improvement over linear regression, yet, as expected, it still wasn't as strong as the gradient boosting models.
