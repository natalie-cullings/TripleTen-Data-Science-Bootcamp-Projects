# Instacart Customer Behavior Analysis
#### *Project 2: Exploratory Data Analysis*

## Overview

This project leverages exploratory data analysis (EDA) to uncover insights about customer behavior on Instacart. The analysis reveals insights around when customers place orders, how frequently they place orders, how many products they buy in an order, and what products are most popular (both in general and also in terms of reordered products).

## Table of Contents

1. [Dataset](#dataset)
2. [Objective](#objective)
3. [Analysis Process](#analysis-process)
4. [Requirements](#requirements)
5. [Results](#results)

## Dataset <a id="dataset"></a>

A description of the data sources this project utilizes is provided below:

**Orders**
*instacart_orders.csv: each row corresponds to one order on the Instacart app*
- 'order_id': ID number that uniquely identifies each order
- 'user_id': ID number that uniquely identifies each customer account
- 'order_number': the number of times this customer has placed an order
- 'order_dow': day of the week that the order placed (which day is 0 is uncertain)
- 'order_hour_of_day': hour of the day that the order was placed
- 'days_since_prior_order': number of days since this customer placed their previous order

**Products**
*products.csv: each row corresponds to a unique product that customers can buy*
- 'product_id': ID number that uniquely identifies each product
- 'product_name': name of the product
- 'aisle_id': ID number that uniquely identifies each grocery aisle category
- 'department_id': ID number that uniquely identifies each grocery department category

**Order Products**
*order_products.csv: each row corresponds to one item placed in an order*
- 'order_id': ID number that uniquely identifies each order
- 'product_id': ID number that uniquely identifies each product
- 'add_to_cart_order': the sequential order in which each item was placed in the cart
- 'reordered': 0 if the customer has never ordered this product before, 1 if they have

**Aisles**
*aisles.csv*
- 'aisle_id': ID number that uniquely identifies each grocery aisle category
- 'aisle': name of the aisle

**Departments**
*departments.csv*
- 'department_id': ID number that uniquely identifies each grocery department category
- 'department': name of the department


The dataset is stored as a CSV file named `movies_and_shows.csv`. 

**Note:** The column names given above are reflective of their names *after* they were renamed in the Jupyter Notebook.

## Objective <a id="objective"></a>

The primary objective of this project is to uncover insights into the shopping habits of Instacart customers.

These data sources were originally published by Instacart on Kaggle for a public competition. 

## Analysis Process <a id="analysis-process"></a>

The analysis involves the following steps:

1. **Data Loading and Exploration**:
   - Import and explore the datasets.
   
2. **Data Preprocessing**:
   - Handling missing, duplicate, or inconsistent data. 
   
3. **Exploratory Data Analysis**:
   - Group and aggregate data.
   - Plot visualizations of the data distributions (bar charts and histograms) to uncover patterns in Instacart customer behaviors.
   
5. **Insights**:
   - Draw conclusions about Instacart customers' shopping habits and the most popular products ordered.

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

## Results <a id="results"></a>

The exploratory data analysis revealed interesting insights into Instacart customers' shopping behaviors. Visualizing distributions of the data with the Matplotlib library made it a lot easier and clearer to identify these insights.

Bananas are the most popular product ordered and Monday is the most popular day of the week to place orders.

Want to learn more? There's a detailed report on all of the project insights at the bottom of the Jupyter Notebook!


