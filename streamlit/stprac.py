# Import Libraries
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image

# Set default page configuration
st.set_page_config(layout="wide",initial_sidebar_state='collapsed')

# Display Bitcoin logo image
image = Image.open('../Resources/Images/Bitcoin-Logo.png')
st.image(image, width = 600)
st.markdown("# Bitcoin Predictor Alpha")
st.markdown('### A Bitcoin prediction and analysis project')
 
# Initialize Dataframes
bitcoin_df = pd.read_csv(Path('../master.csv')) 