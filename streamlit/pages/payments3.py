import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import hvplot.pandas
import holoviews as hv
from PIL import Image

image = Image.open('../Resources/Images/cryptoheader.jpg')
st.image(image, width=300)
st.markdown("# Select Crypto time ")
st.write('Welcome to the crypto select screen')

st.write('Choose your investment amount')
with st.form("investment_form", clear_on_submit=False):
    user_amount_choice = st.number_input("What amount would you like to invest?")
    st.session_state.user_amount_choice = user_amount_choice
    if user_amount_choice >= 1:
        pass
    else:
        st.error("You must invest more than 1 dollar")
        error = st.form_submit_button("Submit")
        st.stop()
#Submit button
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state.user_amount_choice = user_amount_choice
    st.write(user_amount_choice)
    
    
    #user_amount_choice = st.number_input("What amount would you like to invest?", min_value=1)