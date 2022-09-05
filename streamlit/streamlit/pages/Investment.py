from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
import os
import requests
import pickle
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Intialising Alpaca API
load_dotenv()

alpaca_api_key = os.getenv("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")

api = tradeapi.REST(
    alpaca_api_key,
    alpaca_secret_key,
    'https://paper-api.alpaca.markets/',
    api_version = "v2"
)

# Display image
st.sidebar.image('../Resources/Images/cryptosidebar.png', use_column_width=True)

# User select there investment amount
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

crypto_file_path = Path('../Resources/Top10Crypto.csv')
crypto_df = pd.read_csv(crypto_file_path)
crypto_df = crypto_df['symbol'].dropna()

#User select there crypto
with st.form("crypto_form", clear_on_submit=False):

    user_choice = st.selectbox(
    "Select a single crypto",
    crypto_df
    )

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state.user_choice = user_choice

# Establish Machine Learning Selection



# # User select a Machine Learning Algo
# with st.form("machine_form", clear_on_submit=False):

#     user_choice = st.selectbox(
#     "Select a Machine Learning Model",
#     ml_df
#     )

#     # Every form must have a submit button.
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         st.session_state.user_choice = user_choice

start_date = (datetime.today().date() - timedelta(days=9)).isoformat()
data = api.get_crypto_bars(
    symbol=user_choice,
    timeframe='1Day',
    start=start_date
).df

data = data.loc[data['exchange'] == 'CBSE']
data['sma_fast'] = data['close'].rolling(window=3).mean()
data['sma_slow'] = data['close'].rolling(window=10).mean()
data = data.dropna()

X = data[['sma_fast', 'sma_slow']]

# signal = int()

# # Initiate buying/selling
# if signal == 1:
#     api.submit_order(
#         symbol=crypto,
#         qty=amount,
#         time_in_force='gtc'
#     )
#     print(f'Successfully bought {round(amount, 5)} of {crypto}.')
# elif signal == -1:
#     api.submit_order(
#         symbol=crypto,
#         qty=amount,
#         side='sell',
#         time_in_force = 'gtc'
#     )
#     print(f'Successfully sold {round(amount, 5)} of {crypto}.')



