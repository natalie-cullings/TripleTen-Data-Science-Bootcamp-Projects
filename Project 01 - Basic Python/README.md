# IMDb TV Show Ratings Analysis
#### *Project 1: Basic Python*

## Overview

This project utilizes the basic fundamentals of Python to explore movie and TV show data from the entertainment industry. Its primary objective is to analyze trends during the "Golden Age" of television, which began in 1999 with *The Sopranos* and continues to this day. This analysis examines the TV show data specifically to uncover the relationship between the quantity of user votes and TV show ratings. 

## Table of Contents

1. [Dataset](#dataset)
2. [Objective](#objective)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Dataset <a id="dataset"></a>

The dataset for this project contains various information about movies and TV shows:

- **title:** The title of the movie or TV show. 
- **type:** Specifies if the record is related to a movie or TV show.
- **genres:** A list of genres the movie or TV show is classified under.
- **name:** The first and last name of the prominent person associated with the title (the director or a leading actor).
- **character:** The name of the character played in the title (for actors).
- **role:** Specifies if the given name is an actor's or the director.
- **release_year:** The year that the movie or TV show was released.
- **imdb_votes:** The number of IMDb votes (ratings) a movie or TV show has received.
- **imdb_rating:** The IMDb rating of the movie or TV show. 


The dataset is stored as a CSV file named `movies_and_shows.csv`. 

**Note:** The column names given above are reflective of their names *after* they were renamed in the Jupyter Notebook.

## Objective <a id="objective"></a>

The primary objective of this project is to investigate whether TV shows that received high ratings during the "Golden Age" of television also attracted the most votes from viewers.

## Analysis Process <a id="analysis-process"></a>

The analysis involves the following steps:

1. **Data Loading and Exploration**:
   - Import and explore the dataset.
   
2. **Data Preprocessing**:
   - Handling missing, duplicate, or inconsistent data.Analyzing distributions, trends, and correlations between votes and ratings.
   
3. **Data Filtering**:
   - Filter the dataset to only include TV shows released in 1999 or later.
   - Remove outliers in the dataset.
   
4. **Exploratory Data Analysis**:
   - Group titles into rounded IMDb score buckets and calculate the average number of votes for each grouped score.
   
5. **Insights**:
   - Draw conclusions about viewer preferences and the popularity of shows during the "Golden Age" of television.

## Requirements <a id="requirements"></a>

To run this project, you'll need to install the necessary Python dependencies. These can be installed using `pip` by following the steps below:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

   The `requirements.txt` file includes the following package:
   - `pandas==2.2.3`

## Results <a id="results"></a>

The analysis supports the assumption that highly-rated TV shows released during the "Golden Age" of television also have the most votes. 

Please view the Jupyter Notebook for detailed insights. 