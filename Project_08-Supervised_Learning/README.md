# Project 8: Supervised Learning
#### *Customer Churn Prediction Model*

## Table of Contents

1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Analysis Process](#analysis-process)
5. [Requirements](#requirements)
6. [Results](#results)

## Overview <a id="overview"></a>

The hypothetical company Beta Bank's customers are been increasingly leaving the bank, little by little, each month. The bankers have figured out that it’s cheaper to save the existing customers rather than to attract new ones.

 Using data on clients’ past behavior and termination of contracts with the banks, this supervised learning project predicts whether a customer will leave the bank soon.

## Objective <a id="objective"></a>

Build a model with the maximum possible F1 score. The model's F1 score for the test set must be at least 0.59. In addition, interpret the comparision between the AUC-ROC metric to the F1 score. 

## Dataset <a id="dataset"></a>

This project relies on the provided `Churn.csv` data file. 

The dataset includes the following columns:
- Features
    - RowNumber — the data string index
    - CustomerId — the unique customer identifier
    - Surname — the surname of the customer
    - CreditScore — the credit score of the customer
    - Geography — the customer's country of residence
    - Gender — the customer's gender
    - Age — the customer's age
    - Tenure — the period of maturation for a customer’s fixed deposit (years)
    - Balance — the customer's account balance
    - NumOfProducts — the number of banking products used by the customer
    - HasCrCard — whether or not the customer has a credit card
    - IsActiveMember — whether or not the customer is active
    - EstimatedSalary — the customer's estimated salary
- Target
    - Exited — whether or not the сustomer has left the bank

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and Exploration:**
   - Import and inspect the datasets.

2. **Data Preprocessing:**
    - Drop unnecessary columns, fill missing values in, and encode the categorical variables. Next, define the features and target for the model and study the class balance of the target.

3. **Split the Data into Train, Validate, and Test Sets:**
    - Split the data into training, validation, and testing sets then specify features and a target for each.

4. **Model Optimization:**
    - Optimize the model's hyperparameters.

5. **Model Training:**
    - Train and fit the model. Calculate its F1 score. 
    - Address the class imbalance, re-train and fit the model, and calculate the new F1 score.

6. **Calculate the Testing Set Score:**
   - Calculate the F1 and AUC ROC score of the testing set.

7. **Insights**:
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
   

## Results <a id="results"></a>

Ultimately, the Random Forest Classifier model delivered the performance our project demanded. This model typically performs at a high degree of accuracy because it uses an ensemble of trees instead of just one, which also helps to avoid overfitting. However, as a drawback, it often runs slower compared to a Logistic Regression or Decision Tree model. 

Accuracy is important here because the bank must be able to predict customer churn, and therefore hopefully take the adequate action to prevent it, as accurately as possible. 

With an imbalanced dataset, such as this one, accuracy can be misleading because a model that accurately predicts the majority class can have a high accuracy score but still a low F1 score because is cannot predict the minority class well. 

The F1 score is a more balanced and informative score since it considers both precision (true positives out of predicted positives) and recall (true positive rate), which better measures the model's performance on the minority (positive) class--this is of particular importance in this project. The F1 score is based on a specific threshold whereas the AUC-ROC score considers the performance of the model across all possible thresholds and therefore provies a more robust evaluation of the model.

Check out the project's Jupyter Notebook for much greater detail.
