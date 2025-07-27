# Project 17: Integrated Project #3 (Final Project)
#### *Customer Churn Prediction Model*

## Table of Contents

1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Analysis Process](#analysis-process)
5. [Requirements](#requirements)
6. [Results](#results)
6. [Actionable Recommendations for Interconnect](#recommendations)


## Overview <a id="overview"></a>

The hypothetical telecom operator Interconnect would like to be able to forecast their churn of clients. If a client seems likely to churn, they will be offered promotional codes and special plan options in an effort to retain them.

Interconnect's marketing team has collected some of their clientele's personal data including information about their internet/phone plans and contracts.

Interconnect mainly provides two types of services:
- Landline communication: The telephone can be connected to several lines simultaneously.
- Internet: The network can be set up via a telephone line (DSL, digital subscriber line) or through a fiber optic cable.

Some other services the company provides include:
- Internet security: antivirus software (DeviceProtection) and a malicious website blocker (OnlineSecurity)
- A dedicated technical support line (TechSupport)
- Cloud file storage and data backup (OnlineBackup)
- TV streaming (StreamingTV) and a movie directory (StreamingMovies)

The clients can choose a month-to-month, 1-year, or 2-year contract. 

They can use various payment methods and receive an electronic invoice after a transaction.

## Objective <a id="objective"></a>

The objective of this project is to build a model that will predict customer churn for Interconnect. 
The test set model must produce an AUC-ROC ≥ 0.88 to achieve the highest level of difficulty for the project.

## Dataset <a id="dataset"></a>

This project relies on the following data files:
- `contract.csv` — Information about the contract.
- `personal.csv` — Personal information about the client.
- `internet.csv` — Information about the internet services the client has.
- `phone.csv` — Information about the telephone services the client has.

In each file, the column customerID contains a unique code assigned to each client.

The contract information is valid as of February 1, 2020.

Please find an overview of the fields within each data file below.

**contract.csv**
 - **customerID:** The unique customer identifier (ID).
 - **BeginDate:** The contract start date.
 - **EndDate:** The contract end date.
 - **Type:** The contract type (Month-to-month, One Year, Two year).
 - **PaperlessBilling:** Whether the customer uses paperless billing or not.
 - **PaymentMethod:** The customer's payment method.
 - **MonthlyCharges:** The total monthly charges (USD) for the customer.
 - **TotalCharges:** The total lifetime charges (USD) for the customer.
 
**personal.csv**
 - **customerID:** The unique customer identifier (ID).
 - **gender:** The gender of the customer.
 - **SeniorCitizen:** Whether the customer is a senior citizen or not.
 - **Partner:** Whether the customer has a partner or not.
 - **Dependents:** Whether the customer has dependents or not.
 
**internet.csv**
 - **customerID:** The unique customer identifier (ID).
 - **InternetService:** The type of internet service the customer has (DSL or Fiber optic).
 - **OnlineSecurity:** Whether the customer has the Online Security service or not.
 - **OnlineBackup:** Whether the customer has the Online Backup service or not.
 - **DeviceProtection:** Whether the customer has the Device Protection service or not.
 - **TechSupport:** Whether the customer has the Tech Support service or not.
 - **StreamingTV:** Whether the customer has the Streaming TV service or not.
 - **StreamingMovies:** Whether the customer has the Streaming Movies service or not.

**phone.csv**
 - **customerID:** The unique customer identifier (ID).
 - **MultipleLines:** Whether the customer has multiple phone lines or not.
 

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and Cleaning:**
   - Load, preview, and clean each dataset.
   
2. **EDA and Feature Engineering:**
   - Perform EDA (Exploratory Data Analysis) by exploring the distribution of features in each dataset.
   - Merge the dataset DataFrames into a single Dataframe for additional EDA based on the integrated data.
   - Engineer new features.
   
3. **Data Preprocessing:**
   - Pre-process the data for model training.
   - Encoding categorical data.
   - Scale numerical features (for features that don't already have binary values).

4. **Split the Data into Training, Validation, and Testing Sets**
 - Specifying the features and target.
 - Split the data into training, validation, and testing sets.

5. **Optimize and Train the Model**:
- Hypertune and train the model on various classifier models including LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, and LGBMClassifier.
- Account for class imbalance in the target.

6. **Evaluate the Model on the Validation Set**:
- Get the AUC-ROC value and accuracy score for each model for the validation set. 

7. **Evaluate the Model on the Test Set**:
   - Take the model with the best score and get the AUC-ROC value and accuracy score for the testing set.
   - Make adjustments as necessary to achieve an AUC-ROC score >= 0.88 (the highest difficulty score for the project).
      - Possible adjustments include handling the class imbalance in different ways and engineering additional features.

8. **Insights**:
   - Make conclusions about the model performance and what it means for Interconnect's business.
   - Reflect on the greatest challenges that the project presented and how they were navigated.

9. **Recommendations for Interconnect**:
   - Make actionable recommendations for Interconnect to reduce customer churn based on the interpretation of, and learnings from, both the model and the overall project process.

## Requirements <a id="requirements"></a>

To run this project, you'll need to install the necessary Python dependencies. These can be installed using `pip` by following the steps below:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

The `requirements.txt` file includes the following packages:
 - `lightgbm==4.5.0`
 - `matplotlib==3.8.4`
 - `numpy==2.1.3`
 - `pandas==2.2.3`
 - `scikit_learn==1.5.2`
 - `seaborn==0.13.2`
 - `shap==0.46.0`

## Results <a id="results"></a>

The main difficulty of this project was understanding what the data in the contract dataset represented. I explored the data, made calculations, and checked individual examples to check the logic behind my assumptions. In a real business setting, we would confirm these assumptions with colleagues and/or stakeholders and would likely have the ability to collect more context about the dataset.

Another challenge of the project was learning how to interpret the feature importance for the model and its business implications for Interconnect.

Some of the key steps within the project were engineering new features from the available data, accounting for class imbalance in the model training, and hypertuning the model parameters for optimal performance.
    - Spending sufficient time exploring the distribution of the data, cleaning the data, and better understanding the realtionships and correlations between the features (pre-existing and engineered) helped make the model training and selection process much easier. The quality and relevance of the features that the model is trained on really matters a lot.

The highest-performing model was the LGBM Classifier model and it achieved a 0.89 AUC-ROC score on the testing set along with an accuracy score of 0.80.
    - LightGBM is a gradient boosting framework that uses tree based learning algorithms. A traditional gradient boosting decision tree boosting type was utilized.


## Actionable Recommendations for Interconnect <a id="recommendations"></a>

The customer churn model project for Interconnect was a great example of how machine learning models can identify customers who are likely to churn with great accuracy, provide insight into factors that drive churn, and therefore also yield actionable insights that could help to further reduce future customer churn.

Thanks to feature importance analysis, the following conclusions about customer churn drivers can be made:
- The longer a customer has been an Interconnect customer, the less likely they are to churn.
- Customers with higher monthly charges are more likely to churn.
- Customers with a two year contract type are the least likely to churn followed by customers with a one year contract type. (Therefore, customers with a month-to-month contract type are more likely to churn.)
- Customers with the fiber optic internet service are more likely to churn.
- The longer a customer has been an Interconnect customer, the less likely they are to churn.

Since Interconnect wants to offer promotional codes and special plan options to customers who are most likely to churn. Here are the most important recommendations Interconnect should implement to help reduce churn:
- Investigate customer dissatisfaction with fiber optic internet and try to fix or improve the problems with it. 
    - While this process is happening, offer customers on fiber optic internet promotions or entice them to switch to DSL internet instead.
- Give customers with month-to-month contracts promotions to get them to switch to one or two year contract types.
- Offer promotions to lower the monthly charges for newer and/or month-to-month contract customers. 
- Strucure monthly pricing in a way that minimizes the occurence of overage charges or price increases.
- Offer new or month-to-month contract type customers a promotional incentive to remain a customer for multiple subsequent months (Further research is needed to understand a meaningful threshold in the relationship between customer tenure and churn status.).
- Conduct customer outreach to new customers and month-to-month customers to better understand the areas of their service that they are happy or unhappy with for additional, actionable insights.



