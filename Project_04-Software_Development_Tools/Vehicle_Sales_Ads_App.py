'''
This script creates a Streamlit dashboard for exploring vehicle sales ads data.
'''

# load necessary libraries
import pandas as pd
import numpy as np
import nbformat as nbf
import matplotlib.pyplot as plt
import plotly.express as px

import streamlit as st

# read data info df
df = pd.read_csv('./csv_files/vehicles_us_cleaned.csv', parse_dates=['date_posted'])

# define the desired column order
column_order = ['make', 'model', 'type', 'model_year', 'price', 'odometer', 'condition', 'paint_color', 'cylinders', 'fuel', 'transmission', 'date_posted', 'days_listed']

# reindex the DataFrame with the desired column order
df = df.reindex(columns=column_order)

# rename the columns for better readability
df.columns = ['Make', 'Model', 'Type', 'Model Year', 'Price', 'Odometer', 'Condition', 'Paint Color', 'Cylinders', 'Fuel', 'Transmission', 'Date Posted', 'Days Listed']

# provide a header and sub-header to indicate the introduction of the dashboard
st.header('Exploring Vehicle Sales Ads Data', anchor='intro')
st.subheader('An Interactive Streamlit Web App Dashboard', divider='violet')

# let users know the checkbox filters the data
st.write('Dashboard Filters:')

# create a checkbox for filtering based on whether the vehicle is new or not
is_new = st.checkbox('Display New Vehicles Only', value=False)

# filter the DataFrame based on the is_new checkbox
if is_new:
    df = df[df['Condition'] == 'New']
else:
    df = df

# create a multiselect for filtering based on vehicle make
make = st.multiselect('Select Vehicle Make:', df['Make'].unique())

# filter the DataFrame based on the make multiselect
if make:
    df = df[df['Make'].isin(make)]
else:
    df = df

# create a multiselect for filtering based on vehicle model
model = st.multiselect('Select Vehicle Model:', df['Model'].unique())

# filter the DataFrame based on the make multiselect
if make:
    df = df[df['Model'].isin(model)]
else:
    df = df


# introduce the graphs section
st.header('Graphs', anchor='graphs')    

# create a scatter plot of price by model year using plotly.express
fig = px.scatter(df, 
                 x='Model Year', 
                 y='Price', 
                 color='Condition',
                 title='Price by Model Year'
                 )


# plot the price by 'Model Year' scatterplot via streamlit
st.plotly_chart(fig, use_container_width=True)

# plot a histogram of 'Model Year'
fig_my = px.histogram(df, 
                   x='Model Year', 
                   title='Histogram of Model Year',
                   labels={'count': 'Count'}
                  )

# update the y-axis title
fig_my.update_yaxes(title_text='Number of Listings')

# plot the histogram of model_year via streamlit
st.plotly_chart(fig_my, use_container_width=True)

# provide a header to indicate the data table follows
st.header('Data Table', anchor='data-table')

# remove commas from display in st.write(df)
df['Model Year'] = df['Model Year'].astype(str)
df['Model Year'] = df['Model Year'].str.replace(',', '')

# remove the timestamp part of the date_posted column
df['Date Posted'] = df['Date Posted'].dt.date

# display the DataFrame
st.dataframe(df)
