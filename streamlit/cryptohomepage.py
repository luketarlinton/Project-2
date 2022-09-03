# Import Libraries
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from numpy import nan
import hvplot.pandas
import holoviews as hv
from PIL import Image

# Set default page configuration
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

# Display Bitcoin logo image
image = Image.open('../Resources/Images/cryptocalculator.jpg')
st.image(image, width = 200)
st.markdown("# Crypto Predictor Alpha")
st.markdown('### A Crypto prediction and analysis project')
st.markdown('Welcome to the homepage of our predictor app')
 
# Initialize Dataframes
bitcoin_df = pd.read_csv(Path('../master.csv'),infer_datetime_format=True, parse_dates=True, index_col='timestamp')

# Structure Data
#bitcoin_df.sort_values()

# sorting by daily % increase after converting 'actual_returns' column to sortable value
copied_bitcoin = bitcoin_df.copy()
copied_bitcoin['Price'] = copied_bitcoin['close'].astype(float)
sorted_bitcoin = copied_bitcoin.sort_values('Price', ascending=True)

daily_plot = sorted_bitcoin.hvplot.area(
    title = 'Daily price of Bitcoin 2015 to 2021',
    stacked = True,
    x= 'timestamp',
    xlabel = 'Date',
    #rot = 45,
    y = 'Price',
    ylabel = 'Price',
    #legend= 'top',
    ylim = (0,80000),
    height=400, 
    width=800,
    alpha = 0.6,
)
st.session_state.daily_plot = daily_plot
st.bokeh_chart(hv.render(daily_plot))

vol_bitcoin = bitcoin_df.copy()
#vol_bitcoin['Price'] = copied_bitcoin['close'].astype(float)
#sorted_vol_bitcoin = vol_bitcoin.sort_values('Price', ascending=True)

vol_plot = vol_bitcoin.hvplot.area(
    title = 'Volume of Bitcoin 2015 to 2021',
    stacked = True,
    x= 'timestamp',
    xlabel = 'Date',
    #rot = 45,
    y = 'volume',
    ylabel = 'Price',
    #legend= 'top',
    ylim = (0,80000),
    height=400, 
    width=800,
    alpha = 0.6,
)
st.session_state.daily_plot = vol_plot
st.bokeh_chart(hv.render(vol_plot))