# Import Libraries
from cgi import test
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import hvplot.pandas
import holoviews as hv
from PIL import Image

# Set default page configuration
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

# Display image
image = Image.open('../Resources/Images/cryptoheader.jpg')
st.image(image, width = 450)
st.markdown("# Crypto Predictor Alpha")
st.markdown('### A Crypto prediction and analysis project')
st.sidebar.image('../Resources/Images/cryptosidebar.png', use_column_width=True)

# Read master file
crypto_file_path = Path('../Resources/Top10Crypto.csv')
crypto_df = pd.read_csv(crypto_file_path)
crypto_df = crypto_df['symbol'].dropna()

st.write('This study was developed to showcase the effectiveness of Machine Learning Algorithms on launching an effective trading strategy with popular cryptocurrencies.')