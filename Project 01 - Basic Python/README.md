# Basic Python Project: Entertainment Data Analysis

## Overview

This project explores data from the entertainment industry, focusing on movies and TV shows. The primary objective is to analyze trends during the "Golden Age" of television, which began in 1999 with *The Sopranos* and continues today. By examining how the number of votes impacts ratings, this analysis aims to uncover patterns associated with high-quality TV shows from this era.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Objective](#objective)
4. [Analysis Process](#analysis-process)
5. [Requirements](#requirements)
6. [How to Run](#how-to-run)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction <a id="introduction"></a>

In this project, we work with entertainment data to examine relationships between user votes and ratings for TV shows. This analysis excludes movies to focus specifically on trends in television.

## Dataset <a id="dataset"></a>

The dataset used in this project contains records of movies and TV shows, including details like:

- Title
- Release Year
- Genre
- Number of Votes
- User Ratings

The dataset is stored as a CSV file named `movies_and_shows.csv`.

## Objective <a id="objective"></a>

The primary aim of this project is to investigate whether TV shows that received high ratings during the "Golden Age" of television also attracted the most votes from viewers.

## Analysis Process <a id="analysis-process"></a>

The analysis involves the following steps:

1. **Data Loading and Cleaning**:
   - Importing the dataset and handling missing or inconsistent data.
   
2. **Exploratory Data Analysis (EDA)**:
   - Analyzing distributions, trends, and correlations between votes and ratings.
   
3. **Data Filtering**:
   - Isolating TV shows and removing irrelevant data such as movies.
   
4. **Visualization**:
   - Creating plots to illustrate relationships between ratings and votes.
   
5. **Insights**:
   - Drawing conclusions about viewer preferences and the popularity of shows during the "Golden Age" of television.

## Requirements <a id="requirements"></a>

The following libraries are used in this project:

- `pandas` for data manipulation
- `matplotlib` for visualization

Install the dependencies using:

```bash
pip install pandas matplotlib
