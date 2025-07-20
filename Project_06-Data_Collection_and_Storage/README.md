# Project 6: Data Collecton and Storage (SQL and HTTP Web Requests)
#### *Chicago Taxi Market Analysis*

## Table of Contents

1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Analysis Process](#analysis-process)
5. [Requirements](#requirements)
6. [Results](#results)

## Overview <a id="overview"></a>

The hypothetical, new ride-sharing company Zuber is launching in Chicago. The company wants to understand the local taxi market and test a hypothesis about the impact of weather on rides.

The weather data for the analysis is obtained via an HTTP request using the get() method. From there, the obtained HTML is parsed using the BeautifulSoup library to get the needed information in the desired format. The csv files were obtained via an export of the result set of SQL queries written against Zuber's database.

## Objective <a id="objective"></a>

The objective is to understand the top taxi companies operating in the area, the most common neighborhoods that customers take taxis to, and to test the hypotheses as to whether the average duration of rides to the airport from a sample neighborhood (the Loop neighborhood) is different on rainy Saturdays compared to the entire dataset. 

## Dataset <a id="dataset"></a>

The project relies on the following three csv files:

**Trips by Taxi Company**
*trips_by_taxi_company.csv: This file contains data around the number of taxi rides per taxi company from November 15-16, 2017.*
- *company_name:* the name of the taxi company
- *trips_amount:* the number of trips conducted by each taxi company

**Average Trips by Destination Neighborhood**
*avg_trips_by_destination_neighborhood.csv: This file contains data around the average number of trips per neighborhood that taxi rides ended in in November 2017.*
- *dropoff_location_name:* the name of the neighborhood that the taxi ride ended in
- *average_trips:* the average number of trips that ended in the specified neighborhood

**Loop to O'Hare Trips**
*loop_to_ohare_trips.csv: This file contains data on rides from the Loop neighborhood to O'Hare International Airport.*
- *start_ts:* the datetime value of when the taxi ride started
- *weather_conditions:* the value of this field is 'Bad' if the weather was rainy or stormy, but 'Good' otherwise
- *duration_seconds:* the duration of the taxi ride in seconds

## Analysis Process <a id="analysis-process"></a>

The analysis process is divided into the following steps:

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

## Requirements <a id="requirements"></a>

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
   

## Results <a id="results"></a>

The taxi company market in the Chicago area is very competitive. Sixteen taxi companies accounted for 75% of trip volume for the analyzed sample while one company, Flash Cab, led the market in trip volume by a signifcant lead. The dataset for trips by taxi company is heavily right-skewed--the relatively higher performance of top companies drives this behavior. 

The most popular neighborhoods that customers took taxis to were the Loop, River North, Streeterville, and West Loop neighborhoods. This data is also heavily right-skewed with the top ten drop-off neighborhoods greatly exceeding the 75th percentile of average trips values. We can hypothesize that certain neighborhoods are larger/more population dense, wealthier, or harder to access via public transportation than the majority of other neighborhoods with a much lower number of average trips.

Lastly, we can confirm that the average trip duration on rainy Saturdays from the Loop neighborhood to O'Hare International Airport was different than that of all trips with the same trip start and end points. The duration of rides (in seconds) was higher at a statistically significant result on rainy Saturdays at 2,427.2 seconds vs. the overall average of 2,071.7 seconds (about a 6 minute difference). Bad weather coupled with a busy travel day and city traffic is no joke! 

Check out the project's Jupyter Notebook for analysis in much greater detail.
