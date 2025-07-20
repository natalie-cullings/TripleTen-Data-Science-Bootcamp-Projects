# Project 5: Integrated Project #1
#### *Identify Patterns that Drive the Success of Video Games*

## Overview

This project involves an analysis of video game sales using Python. The primary goal of this analysis is to explore various factors affecting video game popularity, with a particular focus on understanding the correlations between features such as genre, platform, and global sales. The project aims to provide insights that could be used by game developers and publishers to better understand market trends and consumer preferences.

## Table of Contents

1. [Dataset](#dataset)
2. [Objective](#objective)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Dataset

This project relies on the provided `games.csv` data file, which contains data spanning back to 2016. Note that data for 2016 may be incomplete.

The dataset includes the following columns:

- **name**: The title of the video game.
- **platform**: The gaming platform on which the game was released (e.g., PS4, Xbox One, PC).
- **year_of_release**: The year the game was released.
- **genre**: The genre of the game (e.g., Action, Sports, RPG).
- **publisher**: The company that published the game.
- **na_sales**: Sales in North America (in millions of units).
- **eu_sales**: Sales in Europe (in millions of units).
- **jp_sales**: Sales in Japan (in millions of units).
- **other_sales**: Sales in other regions (in millions of units).
- **global_sales**: Total worldwide sales (in millions of units).


## Objective

The main objective of this project is to analyze video game sales and identify which features are most influential in determining a game's success. The analysis aims to build a clear understanding of how factors such as platform, genre, and release year impact sales.

## Analysis Process

The analysis process is divided into the following steps:

1. **Data Loading and Exploration**:
   - Import and explore the dataset, and ensure all features are properly formatted for analysis.
   
2. **Data Preprocessing**:
   - Handle missing values, remove duplicates, and manage inconsistencies.

3. **Feature Analysis**:
   - Calculate statistical measures for each feature and explore their distribution.
   - Correlate platform, genre, and other attributes with global sales.
   
4. **Exploratory Data Analysis**:
   - Create visualizations such as histograms, scatter plots, and box plots to analyze relationships between features and global sales.

5. **Insights**:
   - Draw conclusions about which factors most strongly impact video game sales and propose potential strategies for developers and publishers.

## Requirements

To run this project, the necessary Python dependencies can be installed with `pip`. Please follow these steps:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
   The `requirements.txt` file includes the following packages:
   - `pandas==2.2.3`
   - `matplotlib==3.8.4`
   - `numpy==2.1.3`
   - `scipy==1.14.1`

## Results

The analysis revealed several interesting insights regarding video game sales. Among them, we found that games released on popular platforms such as PlayStation and Xbox tend to have higher sales. Additionally, certain genres like Action and Sports are consistently top performers in terms of sales. 

For detailed visual insights and statistical findings, please view the attached Jupyter Notebook, which includes all plots, correlation metrics, and a more thorough explanation of the methodology.

Hopefully these insights can assist game developers and publishers in making informed decisions about game development, marketing, and release strategies.
