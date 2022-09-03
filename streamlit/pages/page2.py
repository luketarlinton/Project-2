# Import Libraries
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import hvplot.pandas
import holoviews as hv
from PIL import Image
import os
import requests
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Display image
image = Image.open('../Resources/Images/cryptoheader.jpg')
st.image(image, width=300)
st.markdown("# Select Crypto time ")
st.write('Welcome to the crypto select screen')

# Read master file
crypto_file_path = Path('../Resources/Top10Crypto.csv')
crypto_df = pd.read_csv(crypto_file_path)
crypto_df = crypto_df['symbol'].dropna()


with st.form("my_form", clear_on_submit=False):

     user_choice = st.selectbox(
     "Select a single crypto",
     crypto_df
     )

    # Every form must have a submit button.
     submitted = st.form_submit_button("Submit")
if submitted:
    
    # Use API to get data for tickers
    load_dotenv()

    alpaca_api_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
    #alpaca_api_key = "PK6YPZKPVN3VL6DG4F47"
    #alpaca_secret_key = "FXZfqcsmRan79XJEi6JvCzdszk79Srg5jrCmqBAH"

    api = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        api_version = "v1beta2"
    )
    # st.write(type(alpaca_api_key))
    ticker_choice = api.get_crypto_bars(user_choice, TimeFrame.Day, "2021-06-06", "2022-08-08").df
    # ticker_choice_1 = ticker_choice
    # st.write(ticker_list)
    # my_df = pd.DataFrame(ticker_choice)
    # my_df.to_csv('x.csv')
    #st.write(ticker_choice)
    my_df = ticker_choice["close"]
    my_df.columns = user_choice

    #Plot 5 year prices
    my_df_plot = my_df.hvplot.area(title=f'How {user_choice} is performing', ylabel='Price', xlabel='Date',height=500,
                                width=700)
    st.session_state.my_df_plot = my_df_plot
    st.bokeh_chart(hv.render(my_df_plot)) 
    
# Seting up signals
    st.write('Seting up signals')
    data = my_df.to_frame()
    data['actual_returns'] = data.pct_change()
    data['sma_fast'] = data['close'].rolling(window=3).mean()
    data['sma_slow'] = data['close'].rolling(window=10).mean()
    data = data.dropna()
    data['signal'] = 0
    data.loc[(data["actual_returns"] >= 0), "signal"] = 1
    data.loc[(data["actual_returns"] < 0), "signal"] = -1
    X = data[['sma_fast', 'sma_slow']].shift().dropna()
    y = data['signal'][1:]
    st.write(data.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Using Logistic Regression Model (Model 1)
    lr_model = LogisticRegression(random_state=5)
    lr_model.fit(X_train, y_train)

    training_predictions = lr_model.predict(X_train)
    testing_predictions = lr_model.predict(X_test)

    lr_report_train = classification_report(y_train, training_predictions, output_dict=True)
    lr_report_train = pd.DataFrame(lr_report_train).transpose()

    lr_report_test = classification_report(y_test, testing_predictions, output_dict=True)
    lr_report_test = pd.DataFrame(lr_report_test).transpose()
    
    # Using GaussianNB Model (Model 2)
    gauss_model = GaussianNB()
    gauss_model.fit(X_train, y_train)

    gauss_pred = gauss_model.predict(X_train)
    gauss_test = gauss_model.predict(X_test)

    gauss_report_train = classification_report(y_train, gauss_pred, output_dict=True)
    gauss_report_train = pd.DataFrame(gauss_report_train).transpose()

    gauss_report_test = classification_report(y_test, gauss_test, output_dict=True)
    gauss_report_test = pd.DataFrame(gauss_report_test).transpose()

    # Using Decision Trees Model (Model 3)
    tree = DecisionTreeClassifier()
    tree_model = tree.fit(X_train_scaled, y_train)
    
    tree_pred = tree_model.predict(X_train)
    tree_test = tree_model.predict(X_test)

    tree_report_train = classification_report(y_train, tree_pred, output_dict=True)
    tree_report_train = pd.DataFrame(tree_report_train).transpose()

    tree_report_test = classification_report(y_test, tree_test, output_dict=True)
    tree_report_test = pd.DataFrame(tree_report_test).transpose()

    # Using SVC
    

    # Display report section
    tab1, tab2, tab3, tab4 = st.tabs(["Logistic Regression", "Gaussian", "Decision Tree","Model_cat"])

    with tab1:
        tab1.header("Logistic Regression Model")   
        col1, col2 = st.columns(2)

        with col1:
            tab1.subheader("Training Results")
            tab1.write(lr_report_train, width=100)

        with col2:
            tab1.subheader("Testing Results")
            tab1.write(lr_report_test)
        
        tab1.markdown("""---""")
        tab1.subheader("Summary")
        accuracy_lr = lr_report_test.loc["accuracy","f1-score"]
        recall_lr = lr_report_test.loc["macro avg","recall"]
        tab1.write(f"The overall accuracy of the Logistic Regression Model was {round(accuracy_lr,2)}")
        tab1.write(f"The overall accuracy of the Logistic Regression Model was {round(recall_lr,2)}")
    
    with tab2:
        tab2.header("GaussianNB Model")
        #tab2.image("https://static.streamlit.io/examples/dog.jpg", width=200)

        with col1:
            tab2.subheader("Training Results")
            tab2.write(gauss_report_train)

        with col2:
            tab2.subheader("Testing Results")
            tab2.write(gauss_report_test)
        
        tab2.markdown("""---""")
        tab2.subheader("Summary")
        accuracy_gauss = gauss_report_test.loc["accuracy","f1-score"]
        recall_gauss = gauss_report_test.loc["macro avg","recall"]
        tab2.write(f"The overall accuracy of the GaussianNB Model was {round(accuracy_gauss,2)}")
        tab2.write(f"The overall accuracy of the GaussianNB Model was {round(recall_gauss,2)}")
    
    with tab3:
        tab3.header("Decision Tree Classifier Model")
        #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

        with col1:
            tab3.subheader("Training Results")
            tab3.write(tree_report_train)

        with col2:
            tab3.subheader("Testing Results")
            tab3.write(tree_report_test)
        
        tab3.markdown("""---""")
        tab3.subheader("Summary")
        accuracy_tree = tree_report_test.loc["accuracy","f1-score"]
        recall_tree = tree_report_test.loc["macro avg","recall"]
        tab3.write(f"The overall accuracy of the GaussianNB Model was {round(accuracy_tree,2)}")
        tab3.write(f"The overall accuracy of the GaussianNB Model was {round(recall_tree,2)}")

    #with tab4:
