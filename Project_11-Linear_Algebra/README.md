# Project 11: Linear Algebra
#### *Practical Application of Linear Algebra Case Study*

## Table of Contents

1. [Overview & Objective](#overview_objective)
2. [Dataset](#dataset)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Overview & Objective <a id="overview_objetive"></a>

The hypothetical insurance company Sure Tomorrow wants to solve several tasks with the help of machine learning.
- Task 1: Find customers who are similar to a given customer. This will help the company's agents with marketing.
- Task 2: Predict whether a new customer is likely to receive an insurance benefit. Can a prediction model do better than a dummy model?
- Task 3: Predict the number of insurance benefits a new customer is likely to receive using a linear regression model.
- Task 4: Protect clients' personal data without breaking the model from the previous task. 
    - It's necessary to develop a data transformation algorithm that would make it hard to recover personal information if the data fell into the wrong hands. This is called data masking, or data obfuscation. 
    - The data should be protected in such a way that the quality of machine learning models doesn't suffer
    - *The goal isn't to pick the best model, but just to prove that the algorithm works correctly.*


## Dataset <a id="dataset"></a>

This project relies on the provided `insurance_us.csv` data file. 

The dataset includes the following columns:
- Features
    - **gender:** The gender of the insured person.
    - **age:** The age of the insured person.
    - **salary:** The salary of the insured person.
    - **Family members:** The number of family members the insured person has.
- Target
    - **Insurance benefits:** The number of insurance benefits received by the insured person over the last five years.

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and Processing:**
   - Load the data and check that it’s free from obvious issues. 

2. **Exploratory Data Analysis (EDA):**
    - Examine a pair plot of the customer data to better understand customer characteristics.
   
3. **Identify Similar Customers:**
    - Define a function to get the K-nearest neighbors using the Euclidean and Manhattan distances. 
    - Compare how scaled and unscaled data affect the kNN algorithm.

4. **Identify Whether a Customer is Likely to Receive an Insurance Benefit:**
    - Build a binary classification machine learning model to determine whether a customer is likely to receive an insurance benefit or not.
    - Ensure it performs better than simple “dummy” model.

5. **Build a Linear Regression Model:**
   - Build a linear regression model and evaluate what RMSE would be for it. Check for both the scaled and non-scaled data.

6. **Obfuscate the Data:**
   - Obfuscate the data by multiplying the numerical features by an invertible matrix P.

7. **Prove Data Obfuscation Works with Linear Regression:**
   - Analytically prove that the given obfuscation method won't affect linear regression in terms of predicted values (i.e. their values will remain the same). 
   - Prove that Linear Regression can work computationally with the chosen obfuscation transformation. 
    - Compare the predicted values, RMSE, and R-squared metric values between when the model is run on the original data vs. the obfuscated data.

8. **Insights**:
   - Draw conclusions about the project and its outcomes.

## Requirements <a id="requirements"></a>

To run this project, the necessary Python dependencies can be installed with `pip`. Please follow these steps:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
The `requirements.txt` file includes the following packages:
- `imbalanced_learn==0.11.0`
- `ipython==8.12.3`
- `numpy==2.1.3`
- `pandas==2.2.3`
- `pip==24.1.2`
- `scikit_learn==1.5.2`
- `seaborn==0.13.2`
   

## Results <a id="results"></a>

Great news for The Sure Tomorrow insurance company--all the tasks they need help with can be solved using machine learning!

**Task 1:**
Customers who are similar to a given customer can be identified with a K-nearest neighbors (KNN) function. Using scaled data to train the model resulted in different neighbors than when using non-scaled data to train the model. The distances between neighbors were more comparable between euclidean and manhattan distance calculations using the scaled data as well.

**Task 2:**
The KNN model can predict whether a new customer is likely to receive an insurance benefit with great accuracy. The performance of the model blew the performance of the dummy model out of the water. The model performed better when it was trained with scaled data and oversampling our minority positive class greatly improved its F1 score.

**Task 3:**
A linear regression model can predict the number of insurance benefits a new customer is likely to receive. The performance of the model was the same, for the training and testing sets respectively, regardless of whether the model was trained with scaled or non-scaled data.

**Task 4:**
Obfuscation protected clients' personal data without breaking the model from task 3 (or degrading its performance). Additional insights were gained into the properties of matrices, enabling an analytical proof that obfuscating personal data does not affect the model's RMSE.