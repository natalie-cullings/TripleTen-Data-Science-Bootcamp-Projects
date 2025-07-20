# Project 10: Real-World Data Science Problem (Integrated Project #2)
#### *Gold Recovery from Gold Ore Purification Process Prediction & Optimization Model*

## Table of Contents

1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Analysis Process](#analysis-process)
5. [Requirements](#requirements)
6. [Results](#results)

## Overview <a id="overview"></a>

This project covers a practical data science problem from the gold mining sector for [Zyfra](https://www.zyfra.com/), a real company that develops efficiency solutions for heavy industry.  

## Objective <a id="objective"></a>

The objective of this project is to prepare a prototype of a machine learning model for Zyfra that predicts the amount of gold recovered from gold ore and therefore optimizes the production process while simultaneously eliminating unprofitable parameters.

## Dataset <a id="dataset"></a>

The data this project relies on is stored in the three following csv files:
- **gold_recovery_train.csv:** The training dataset.
- **gold_recovery_test.csv:** The test dataset.
- **gold_recovery_full.csv:** The source dataset.

Within each file, the features (columns) are named using the following convention:<br/>
**[stage].[parameter_type].[parameter_name]**

*Possible values for [stage]:*
- **rougher:** The flotation stage.
- **primary_cleaner:** The primary purification stage.
- **secondary_cleaner:** The secondary purification stage.
- **final:** The final characteristics at the end of the process.

*Possible values for [parameter_type]:*
- **input:** The raw material parameters.
- **output:** The product (output) parameters.
- **state:** The parameters that characterize the current state of the stage.
- **calculation:** The calculation characteristics.

*Please see the project Jupyter notebook for a thorough overview of the ore purification process.*


## Analysis Process <a id="analysis-process"></a>

1. Prepare the data.
 - Open the files and look into the data.
 - Check that the gold recovery calculation following the rougher stage is correct by finding the MAE (mean absolute error) between the given rougher.output.recovery feature value and manually calculating its value.
 - Determine which features aren't available in the test set.
 - Clean and preprocess the data.

2. Analyze the data.
 - Explore how the metal concentrations (Au, Ag (Silver), and Pb (Lead)) change after each purification stage.
 - Compare the feed particle size distributions between the training and test sets.
 - Analyze the total concentrations of all substances at the raw feed, rougher concentrate, and final concentrate stages.

 3. Build and evaluate the model.
 - Write a function to calculate the final SMAPE (symmetrical mean absolute percentage error).
 - Train different models and evaluate them using cross-validation. 
 - Test the best model (using the test sample).

1. **Data Loading and Preparation:**
    - Open the files and look into the data.
    - Check that the gold recovery calculation following the rougher stage is correct by finding the MAE (mean absolute error) between the given rougher.output.recovery feature value and manually calculating its value.
    - Determine which features aren't available in the test set.
    - Clean and preprocess the data.
   
2. **Data Analysis:**
 - Explore how the metal concentrations (Au, Ag (Silver), and Pb (Lead)) change after each purification stage.
 - Compare the feed particle size distributions between the training and test sets.
 - Analyze the total concentrations of all substances at the raw feed, rougher concentrate, and final concentrate stages.
   
3. **Build and Evaluate the Model:**
 - Write a function to calculate the final SMAPE (symmetrical mean absolute percentage error).
 - Train different models and evaluate them using cross-validation. 
 - Test the best model (using the test sample).
   
4. **Insights**:
   - Determine the best model and assess its performance.
   - Communicate insights about the performance of the model.

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
- `scikit_learn==1.5.2`

## Results <a id="results"></a>

The Random Forest Regressor model resulted in the best final SMAPE score based on our training set (cross-validated) data. After testing it on the test set, a final SMAPE score of 9.93% was achieved (25% of the SMAPE score of the rougher.output.recovery target and 75% of the SMAPE score of the final.output.recovery target). The final model performs better than a simple constant or baseline model for both the training and testing sets.

The final SMAPE score reflects a model that has a relatively low average error relative to the actual values, which indicates good predictive accuracy. It also indicates that the model's errors are reasonably balanced between underestimating and overestimating. Moving forward, Zyfra can leverage this machine learning model to optimize its gold extraction process and therefore optimize its profits through more efficient extraction and the elimination of unprofitable or costly process parameters.

Please check out the project's Jupyter Notebook for much greater detail.
