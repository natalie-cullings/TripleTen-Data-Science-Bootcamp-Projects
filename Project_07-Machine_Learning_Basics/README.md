# Project 7: Machine Learning Basics
#### *Phone Subscription Plan Recommendation Model*

## Table of Contents

1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Analysis Process](#analysis-process)
5. [Requirements](#requirements)
6. [Results](#results)

## Overview <a id="overview"></a>

The hypothetical mobile carrier company Megaline has found out that many of their subscribers use legacy plans. They want to develop a model that would analyze subscribers' behavior and recommend one of Megaline's newer plans: Smart or Ultra.

## Objective <a id="objective"></a>

Using monthly behaviorial data about subscribers who have already switched to the new plans, the objective of this project is to develop a machine learning model to help pick the right plan for legacy plan users.

Optimize various models that are suitable for a classification task with the goal of producing the highest possible accuracy score for each model. It must have an accuracy score of at least 75%.


## Dataset <a id="dataset"></a>

This project relies on the provided `users_behavior.csv` data file. Every observation in the dataset contains monthly behavior information about one user.

The dataset includes the following columns:

- **сalls**: The number of calls made.
- **minutes**: The total call duration in minutes.
- **messages**: The total number of text messages.
- **mb_used**: The total amount of internet traffic consumed in MB.
- **is_ultra**: The plan for the current month (Ultra = 1, Smart = 0).

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and Exploration:**
   - Import and inspect the datasets.

2. **Split the Data into Train, Validate, and Test Sets:**
    - Split the data into training, validation, and testing sets then specify features and a target for each.
   
3. **Model Training:**
    - Train the Decision Tree Classifier, Random Forest Classifier, and Logistic Regression models using the training data set.

4. **Model Optimization:**
    - Optimize the accuracy score of the validation data set for each model.

5. **Assess the Testing Set Accuracy Score:**
   - Assess the accuracy score of the testing data set for the best instance of each model. Assess the best model to use in order to help Megaline pick the right plan for its legacy plan users.

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
   - `numpy==2.1.3`
   - `pandas==2.2.3`
   - `scikit_learn==1.5.2`
   

## Results <a id="results"></a>

The decision tree classifier model was the best model overall with an accuracy score of 78.22%.

The logistic regression model performed better than the random forest classifier model with an accuracy score of 75.89% vs. 75.73%, respectively. 

All three models beat our project's minimum accuracy score requirement of 75%.

Check out the project's Jupyter Notebook for much greater detail.
