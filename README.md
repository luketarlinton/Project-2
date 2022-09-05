# Bitcoin Trading Bot
### Financial Technology Project 2 Group 3:

## Group Members:
- Angus Clark
- Vima Chen
- Luke Tarlinton
- Michael Blauberg
- Adam Westlake

## Project Aim
Create an optimal machine learning model, that buys and sell cryptocurrency in real time, depending on indicators found using Simple Moving Averages (SMA). The windows for the simple moving averages used were 3 and 10 days. This was done by first analysing multiple machine learning models, by training them around a set of historic Bitcoin data. The most accurate model was the Support Vector Classifcation (SVC) model, which was implemented into the trading bot.

## Usage and Installation Instructions
This application requires the following libraries to function. This was built using the package and environemnt manager, Conda, however, another manage may be used. The following dependencies are required:
- pandas
- alpaca-trade-api
- sklearn
- streamlit

To run the streamlit file, navigate to the folder on terminal, then type:

-> *streamlit run cryptohomepage.py*

To run the trading bot, navigate to the project folder, then type:

-> *python3 bot.py*

Note: the environment with all the dependencies needs to be active.

## Results
In the end, we were able to output a trading bot that checks the trading signal every hour, and buys, sells or holds depending on the signal. We were also able to create an accurate machine learning model and evaluate it against other ones with different classifiers. We created a user-interface, that if given more time, we would have implemented the whole project within.

## Evaluation
If given more time, we’d implement and improve on many different parts. This includes:
- Allowing the bot to invest in multiple cryptocurrencies at once.
- Optimising the models for greater accuracy.
- Adding more features to the interface.
- Allow consumers to view how different cryptocurrencies would have performed, using our model, with historic data - comparing actual returns to the trading algorithm returns.