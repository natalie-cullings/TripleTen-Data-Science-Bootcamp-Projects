# Project 3: Statistical Data Analysis
#### *Telecommmunications Plan Revenue Analysis*

## Overview

The commercial department of the hypothetical telecom operator company Megaline offers its clients two prepaid plans: Surf and Ultimate. Megaline's commercial department wants to know which of the plans brings in more revenue in order to adjust its advertising budget.

This project involves a preliminary analysis of the plans based on a relatively small client selection of 500 clients. The goal is to analyze the clients' behavior and determine what plan brings in the most revenue. 

The dataset from 2018 contains information about who the clients are, where they're from, which plan they are on, the number of calls they made, the number of texts they sent, and the amount of data they consumed via web sessions. There is also data available about what each plan includes and the cost of overages beyond what's included.

## Table of Contents

1. [Dataset](#dataset)
2. [Objective](#objective)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Dataset <a id="dataset"></a>

A description of the data sources this project utilizes is provided below:

**Users**
*megaline_users.csv: data on users**
 - **user_id:** The unique user identifier.
 - **first_name:** The user's first name.
 - **last_name:** The user's last name.
 - **age:** The user's age (in years).
 - **reg_date:** The user's subscription date (dd, mm, yy).
 - **churn_date:** The date the user stopped using the service (if the
 value is missing, the calling plan was being used when this database was extracted).
- **city:** The user's city of residence.
- **plan:** The plan name the user is subscribed to.


**Calls**
*megaline_calls.csv: data on calls*
 - **id:** The unique call identifier.
 - **call_date:** The call date.
 - **duration:** The call duration (in minutes).
 - **user_id:** The identifier of the user making the call.

**Messages**
*megaline_messages.csv: data on texts*
 - **id:** The unique text message identifier.
 - **message_date:** The text message date.
 - **user_id:** The identifier of the user sending the text.
 
**Internet**
*megaline_internet.csv: data on web sessions*
 - **id:** The unique web session identifier.
 - **mb_used:** The volume of data spent during the session (in megabytes).
 - **session_date:** The web session date.
 - **user_id:** The identifier of the user conducting the web session.

**Plans**
*megaline_plans.csv: data on the plans*
 - **plan_name:** The calling plan name.
 - **usd_monthly_fee:** The monthly charge of the plan in US dollars.
 - **minutes_included:** The plan's monthly minute allowance.
 - **messages_included:** The plan's monthly text allowance.
 - **mb_per_month_included:** The plan's data volume allowance (in megabytes).
 - **usd_per_minute:** The price per minute after exceeding the package limits (e.g., if the package includes 100 minutes, the 101st minute will be charged).
 - **usd_per_message:** The price per text after exceeding the package limits.
 - **usd_per_gb:** The price per extra gigabyte of data after exceeding the package limits (1 GB = 1024 megabytes).

### **Plan Descriptons**

#### *Surf:*
- The monthly charge is $20.
- It includes 500 monthly minutes, 50 texts, and 15 GB of data.
- After exceeding the package limits...
 - each additional minute costs 3 cents.
 - each additional text message costs 3 cents.
 - each additional GB of data costs $10.


#### *Ultimate:*
- The monthly charge is $70.
- It includes 3,000 monthly minutes, 1,00 texts, and 30 GB of data.
- After exceeding the package limits...
 - each additional minute costs 1 cent.
 - each additional text message costs 1 cent.
 - each additional GB of data costs $7.
 <br>
 
 #### **NOTE:** 
 - Megaline rounds seconds up to minutes, and megabytes to gigabytes. For calls, each individual call is rounded up; even if the call lasted just one second, it will be counted as one minute. 
 - For web traffic, individual web sessions are not rounded up. Instead, the total for the month is rounded up. If someone uses 1025 megabytes this month, they will be charged for 2 gigabytes.

## Objective <a id="objective"></a>

The primary objective of this project is to utilize statistics to establish which Megaline prepaid plan (Surf or Ultimate) brings in more revenue for the company so the commercial department can adjustment its advertising budget accordingly.

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and Exploration:**
   - Import and explore the datasets.
   
2. **Data Preprocessing:**
   - Handling missing, duplicate, or inconsistent data. 
   - Enrich the data as needed.
   
3. **Exploratory Data Analysis:**
   - Merge the DataFrames together.
   - Calculate the monthly revenue of users per month based on the plan's fixed monthly cost and potential usage overage charges.
   - Leverage statistics, including the mean, variance, and standard deviation, to understand how users' monthly usage of average call duration, total minutes spent on calls, total internet traffic consumed, and total messages sent may differ between plans.
   - Plot histograms and box plots to visualize the frequency and distribution of values from the previous step to further assess how users behavior between both plans differ.

4. **Hypothesis Testing**
 - Formulate null and alternate hypotheses about how the population samples of the two plans relate to one another.
 - Conduct the hypothesis tests and interpret the results.
   
5. **Insights**:
   - Explain the results of the analysis in a conclusion and make a recommendation to Megaline's commercial department to guide their advertising budget adjustment decisions.
   - Recommend additional actions based on the project conclusions.

## Requirements <a id="requirements"></a>

To run this project, you'll need to install the necessary Python dependencies. These can be installed using `pip` by following the steps below:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

The `requirements.txt` file includes the following packages:
 - `matplotlib==3.8.4`
 - `numpy==2.1.3`
 - `pandas==2.2.3`
 - `scipy==1.14.1`
 - `seaborn==0.13.2`

## Results <a id="results"></a>

Thanks to our hypothesis tests and applied statistical analysis, we can be confident that users on the Ultimate plan bring in more monthly revenue on average. 

However, since the commercial department has requested this information in the hopes of increasing revenue for the entire population, we should always consider if the requested data is fulfilling their high-level objective. 

Our analysis revealed that client usage behavior between the two plans was actually quite similar. The Surf plan's primary strength is the relative quantity of clients on the plan while the Ultimate plan's primary strength is its higher average monthly revenue. 

Check out the project's Jupyter Notebook for analysis in much greater detail and to learn why we recommended further analysis to inform the commercial department's advertising budget decisions.


