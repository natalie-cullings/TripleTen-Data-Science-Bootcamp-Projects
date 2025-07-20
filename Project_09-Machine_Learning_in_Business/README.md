# Project 9:  Machine Learning in Business
#### *Oil Well Reserve Volume Prediction Model & Profit Optimization*

## Table of Contents

1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Analysis Process](#analysis-process)
5. [Requirements](#requirements)
6. [Results](#results)

## Overview <a id="overview"></a>

The hypothetical oil mining company OilyGiant wants to determine the best place for a new oil well. Parameters of eaach oil well in the region are already known and data on oil samples across three different regions is available.

Choosing the best location for a new oil well involves collecting the oil well parameters (oil quality and volume of reserves) in the selected region, building a model to predict the volume of reserves in the new wells, selecting the oil wells with the highest estimated values, and ultimately picking the region with the highest total profit for the selected oil wells.

## Objective <a id="objective"></a>

Build a machine learning model that will help to pick the region with the highest profit margin. Analyze potential profit and risks using the Bootstrapping technique.

## Dataset <a id="dataset"></a>

A description of the data sources this project utilizes is provided below.

Geological exploration data for the three regions are stored in the following csv files:
- geo_data_0.csv
- geo_data_1.csv
- geo_data_2.csv

Each dataset contains the following fields:
- id — The unique oil well identifier.
- f0, f1, f2 — Three features of potential oil well points. Their specific meaning is unimportant, but the features themselves are significant.
- product — The volume of reserves in the oil well (in thousands of barrels).

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and Preprocessing:**
   - Import and prepare the datasets.

2. **Split the Data into Training and Validation Sets:**
    - Designate the features and target for the model. 
    - Split the data into training and validation sets for each region.

3. **Feature Scaling:**
    - Scale the features for each model.

4. **Model Training:**
    - Train the models and make predictions.

5. **Evaluate the Models' RSME Scores:**
   - Evaluate the models' RSME scores & compare them to the RSME score of each geo's mean target value.

6. **Profit and Risk Calcuations:**
    - Prepare the data for the profit calculation.
    - Calculate the profit and risk for the best 200 well locations in each geo and recommend the best region for development.

7. **Insights**:
   - Draw conclusions about the project and its results.

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

Geo1 is where OilyGiant should develop new oil wells. It has the highest mean profit of $5.2M, the highest range of profit values for its 95% confidence interval range, and it's also the only region with less than a 2.5% probability of realizing a loss, which is OilyGiant's maximum tolerance for risk.

To explore a region, the oil company studies 500 potential oil well locations and picks the best 200 to develop. The bootstrapping technique simulates these studies by taking 1,000 samples of 500 potential oil sites, selecting the best 200 sites out of the 500 potential sites for each sample and, finally, calculating the profit each sample would generate.

From there, the average profitability of each region was calculated based on its 1,000 sample results, which provided an average value that would be very close to the true population mean. The 1,000 bootstrap samples also provided the 95% confidence interval for the distribution of profit values; this ensures confidence that, for geo1 as an example, there is a 95% probability of generating a range of profit from $5.05M to $5.31 in geo 1 no matter which 500 sites OilyGiant decided to study and develop the top 200 sites within.

Geo0 would have been the best region to develop new oil wells in (with a total predicted profit of $33.2M) based on the linear regression models for each region alone, however the bootstraping technique replicated the same conditions that the company employs when exporing a region i.e. developing the best 200 sites out of a given 500 potential sites that they study. Additionally, the bootstapping technique enabled a recommendation that satisfied the maximum risk tolerance of 2.5% for realizing a loss. 

Ultimately, after applying the bootstrapping technique, geo1 was the only region that met the risk tolerance criteria. Fortunately, it also returned the highest average profit value of about $5.2M USD. With a 95% probability, OilyGiant can feel confident that developing 200 new oil wells in geo1 will bring in a range of $5.05M to $5.31M USD and that there is only about a 0.3% probability of realizing a loss from the initiative. Notably, before applying the bootstrapping technique, the geo1 region's linear regression model had the lowest RMSE value, which seems to align with its lowest risk of realizing a loss due to the lower variability of each site's production volume from the region's production volume mean in both actuality and in terms of our model's predicted production levels.

Check out the project's Jupyter Notebook for much greater detail.
