import os
import sys
import pickle
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from sklearn import svm
from sklearn.preprocessing import StandardScaler

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

# Loaded models
model = pickle.load(open(Path('Resources/model.pkl'),'rb'))
scaler = pickle.load(open(Path('Resources/scaler.pkl'), 'rb'))

# User inputs
os.system('clear')
crypto = input('What is the code of the crypto currency  would you like to buy? ') + 'USD'
investment = int(input('How much would you like to invest in total in USD? '))
investment_increment = investment/20

# Function for intiating buy/sell
def intiate():
    os.system('clear')
    price = (api.get_latest_crypto_bar(crypto, 'CBSE')).c
    amount = investment_increment/price
    print(f'The current price of {crypto} is {price} USD')
    print('Intiating buy/sell...')
    # Pull 10 days recent data
    start_date = (datetime.today().date() - timedelta(days=9)).isoformat()
    data = api.get_crypto_bars(
        symbol=crypto,
        timeframe='1Day',
        start=start_date
    ).df
    # Clean and manipulate data to run through model
    data = data.loc[data['exchange'] == 'CBSE']
    data['sma_fast'] = data['close'].rolling(window=3).mean()
    data['sma_slow'] = data['close'].rolling(window=10).mean()
    data = data.dropna()
    # Select and scale X variable
    X = data[['sma_fast', 'sma_slow']]
    X_scaled = scaler.transform(X)
    # Predict the y variable
    signal = model.predict(X_scaled)
    # Initiate buying/selling
    if signal == 1:
        api.submit_order(
            symbol=crypto,
            qty=amount,
            time_in_force='gtc'
        )
        print(f'Successfully bought {round(amount, 5)} of {crypto} for 50 USD.')
    elif signal == -1:
        api.submit_order(
            symbol=crypto,
            qty=amount,
            side='sell',
            time_in_force = 'gtc'
        )
        print(f'Successfully sold {round(amount, 5)} of {crypto} for 50 USD.')

# Start of loop
while True:
    intiate()
    i = 24
    while i > 0:
        print(f'About {i} hours until next buy/sell...')#, end='\r')
        i -= 1
        time.sleep(2)
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
