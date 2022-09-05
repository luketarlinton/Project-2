# "Bitcoin Predictor"
## Financial Technology Project 2 Group 3:

### Group Members:
1. Angus Clark
2. Vima Chen
3. Luke Tarlinton
4. Michael Blauberg
5. Adam Westlake

### Project Aim
Create an optimal machine learning model, that buys and sell cryptocurrency in real time, depending on indicators found using Moving Average Convergence/Divergence (MACD). The windows for the simple moving averages used were 3 and 10 days. This was done by first analysing multiple machine learning models, by training them around a set of historic Bitcoin data. The most accurate model was the Support Vector Classifcation (SVC) model, which was implemented into the trading bot.

### Applying ML in the context of technologies learned:
Utilizing ML and classifiers the objective is to provide analysis of the actual historical returns data against multiple strategy returns with visualization aswell as informing the customer of the perceived confidence (accuracy) with which each ML classifier completed its final analysis.

### Usage and Installation Instructions
This application requires the following libraries to function. This was built using the package and environemnt manager, Conda, however, another manage may be used. The following dependencies are required:
- pandas
- alpaca-trade-api
- sklearn
- streamlit

### Evaluation
If given more time, we’d implement and improve on many different parts. This includes:
- Allowing the bot to invest in multiple cryptocurrencies at once.
- Optimising the models for greater accuracy.
- Adding more features to the interface.
- Allow consumers to view how different cryptocurrencies would have performed, using our model, with historic data - comparing actual returns to the trading algorithm returns.
