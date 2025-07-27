# Project 15: Computer Vision
#### *Convolutional Neural Network Age Prediction Model*

## Table of Contents

1. [Overview & Objective](#overview_objective)
2. [Dataset](#dataset)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Overview & Objective <a id="overview_objective"></a>

The hypothetical supermarket chain Good Seed would like to explore whether Data Science can help them adhere to alcohol laws by making sure they do not sell alcohol to underage people. Their shops are equipped with cameras in the checkout area which are triggered when a person is buying alcohol. 

The project objective is to leverage computer vision methods to build and evaluate a model that can verify the age of customers purchasing alcohol at Good Seed. 

The project notebook was run on Google Collab since the resources of a local machine were not sufficient to run the code.

## Dataset <a id="dataset"></a>

This project relies on the provided `labels.csv` data file and all the jpg image files within the /faces/final_files directory.

Each jpg image is a picture of a person's face. The `labels.csv` file contains the following two fields:
 - **file_name:** The title of the corresponding jpg image within the /faces/final_files directory.
 - **real_age:** The actual age of the person in the corresponding jpg image.

## Analysis Process <a id="analysis-process"></a>

1. **Data Loading and EDA**
   - Load and inspect the data.
    - Make conclusions about the quality of the images by viewing a sample of them.
   - Perform EDA (Exploratory Data Analysis).
    - Check the distribution of photos by age.

2. **Load the Training Set Data and Generate Augmented Image Data**
    - Generate batches of image data with real-time data augmentation using the ImageDataGenerator class in Keras (part of TensorFlow). This helps improve the robustness and generalization of a model by generating altered versions of images during training (i.e. creating more training data from existing samples).

3. **Train the Model**
    - Choose parameters for a neural network model and train the model.

4. **Load the Test Dataset and Evaluate the Model**
    - Load the test dataset.
    - Evaluate the model on the test dataset.

5. **Insights**:
   - Make conclusions about the model's performance and its suitability for Good Seed's use case.

## Requirements <a id="requirements"></a>

To run this project, the necessary Python dependencies can be installed with `pip`. Please follow these steps:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
The `requirements.txt` file includes the following packages:
 - `google-colab==1.0.0`
 - `pandas==1.5.3`
 - `numpy==1.23.5`
 - `matplotlib==3.7.1`
 - `Pillow==9.5.0`
 - `tensorflow==2.12.0`
 - `tensorflow-estimator==2.12.0`
 - `tensorflow-io-gcs-filesystem==0.31.0`
   

## Results <a id="results"></a>

A mean absolute error of 6.5 means the model's predictions are off by 6.5 years on average. This isn't great because the task requires that we have high accuracy to determine a person's age for verification of their legal eligibility to purchase alcohol.

To improve the model, a larger dataset with more diverse examples is needed. The model is currently limited in the distribution of ages present in the dataset, which makes it harder to learn the necessary patterns to predict age more accurately. Factors like a person's pose, facial expression, and any accessories they are wearing (hats, glasses, etc.) increases the model's complexity and reduces its ability to generalize well.

Removing outliers or poor-quality images may reduce confusion within the model. Additionally, balancing the dataset across age groups should help improve data quality.

However, as the model currently stands, it's not ready for the task. The risk of selling alcohol to minors is far too high.