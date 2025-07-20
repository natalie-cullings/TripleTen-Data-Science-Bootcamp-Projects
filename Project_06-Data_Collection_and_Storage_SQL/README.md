# Project 6: Data Collecton and Storage (SQL and HTTP Web Requests/Parsing)
#### *Chicago Ride-sharing Market Analysis*

## Overview

The hypothetical, new ride-sharing company Zuber is launching in Chicago. The company wants to understand the top taxi companies operating in the area, the most common neighborhoods that customers take taxis to, and to test a hypothesis about the impact of weather on rides.

The weather data for the analysis is obtained via an HTTP request using the get() method. From there, the obtained HTML is parsed using the BeautifulSoup library to get the needed information in the desired format. The csv files were obtained via an export of the result set of SQL queries written against Zuber's database. 

## Table of Contents

1. [Dataset](#dataset)
2. [Objective](#objective)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Dataset

The following three csv files are used:
- **trips_by_taxi_company.csv**: this file contains data around the number of taxi rides per taxi company from November 15-16, 2017
- **avg_trips_by_destination_neighborhood.csv**: this file contains data around the average number of trips per neighborhood that taxi rides ended in in November 2017
- **loop_to_ohare_trips.csv**: this file contains data on rides from the Loop neighborhood to O'Hare International Airport

# Data Definitions

**trips_by_taxi_company.csv:**
- *company_name:* the name of the taxi company
- *trips_amount:* the number of trips conducted by each taxi company

**avg_trips_by_destination_neighborhood.csv:**
- *dropoff_location_name:* the name of the neighborhood that the taxi ride ended in
- *average_trips:* the average number of trips that ended in the specified neighborhood

**loop_to_ohare_trips.csv:**
- *start_ts:* the datetime value of when the taxi ride started
- *weather_conditions:* the value of this field is 'Bad' if the weather was rainy or stormy, but 'Good' otherwise
- *duration_seconds:* the duration of the taxi ride in seconds

## Objective

The main objective of this project is to analyze video game sales and identify which features are most influential in determining a game's success. The analysis aims to build a clear understanding of how factors such as platform, genre, and release year impact sales.

## Analysis Process

The analysis process is divided into the following steps:

- Import the csv files
- Study the data they contain
- Ensure the data and data types are as expected
- Identify the top 10 neighborhoods customers take taxi rides to
- Visualize the taxi companies by number of trips
- Visualize the top 10 neighborhoods customers take taxi rides to by average trips
- Draw conclusions about the data
- Test the hypothesis that rides from the Loop neighborhood to O'Hare International Airport are  longer in duration, on average, on rainy Saturdays compared to all trips to the airport from Loop
- Draw conclusions about the hypothesis test

1. **HTTP Web Request for Chicago Weather Data**:
   - Leverage the Requests and BeautifulSoup Python packages to get and parse data hourly data about the weather in Chicago in November 2017.

2. **Query Zuber's Database with SQL to Obtain the Analysis Datasets**:
    - The database used in this project was provided within the TripleTen UI directly and is not available to query within this project, however the project notebook (in the Appendix) contains an overview of the database tables and the SQL used to obtain the datasets.
   
3. **Data Loading and Preprocessing**:
   - Import the csv files and ensure the data types are as expected. Since the datasets were obtained via SQL queries, their contents are already familiar.
   
4. **Exploratory Data Analysis**:
   - Identify the top 10 neighborhoods customers take taxi rides to, visualize the taxi companies by number of trips, and visualize the top 10 neighborhoods customers take taxi rides to by average trips.

 6. **Hypothesis Test**:
   - Test the hypothesis that rides from the Loop neighborhood to O'Hare International Airport are longer in duration, on average, on rainy Saturdays compared to all trips to the airport from Loop.

6. **Insights**:
   - Draw conclusions about the data and the results of the hypothesis test.

## Requirements

To run this project, the necessary Python dependencies can be installed with `pip`. Please follow these steps:

1. Make sure you have Python installed (preferably version 3.6 or higher).
2. Use the following command to install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
   The `requirements.txt` file includes the following packages:
   - `beautifulsoup4==4.12.3`
   - `matplotlib==3.8.4`
   - `pandas==2.2.3`
   - `scipy==1.14.1`
   - `Requests==2.32.3`
   

## Results

The taxi company market in the Chicago area is very competitive. Sixteen taxi companies accounted for 75% of trip volume for the analyzed sample while one company, Flash Cab, led the market in trip volume by a signifcant lead. The dataset for trips by taxi company is heavily right-skewed--the relatively higher performance of top companies drives this behavior. 

The most popular neighborhoods that customers took taxis to were the Loop, River North, Streeterville, and West Loop neighborhoods. This data is also heavily right-skewed with the top ten drop-off neighborhoods greatly exceeding the 75th percentile of average trips values. We can hypothesize that certain neighborhoods are larger/more population dense, wealthier, or harder to access via public transportation than the majority of other neighborhoods with a much lower number of average trips.

Lastly, we can confirm that the average trip duration on rainy Saturdays from the Loop neighborhood to O'Hare International Airport was different than that of all trips with the same trip start and end points. The duration of rides (in seconds) was higher at a statistically significant result on rainy Saturdays at 2,427.2 seconds vs. the overall average of 2,071.7 seconds (about a 6 minute difference). Bad weather coupled with a busy travel day and city traffic is no joke! 

Check out the project's Jupyter Notebook for analysis in much greater detail.
