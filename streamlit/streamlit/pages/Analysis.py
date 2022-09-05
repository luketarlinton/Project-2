# Import Libraries
from cgi import test
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
from sklearn import svm
from pandas.tseries.offsets import DateOffset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Set default page configuration
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

# Display image
image = Image.open('../Resources/Images/cryptocalculator.jpg')
st.image(image, width = 450)
st.markdown("# Crypto Predictor Alpha")
st.markdown('Welcome to the analysis page of our predictor app')
st.sidebar.image('../Resources/Images/cryptosidebar.png', use_column_width=True)

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

    api = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        api_version = "v1beta2"
    )

    # Initialize progress bar
    my_bar = st.progress(0)
    my_bar.progress(1)
    my_bar.progress(2)

    # st.write(type(alpaca_api_key))
    ticker_choice = api.get_crypto_bars(user_choice, TimeFrame.Day, "2015-06-06", "2022-08-08").df
    my_df = ticker_choice["close"]
    my_df.columns = user_choice
    my_df.dropna(inplace=True)

    my_bar.progress(75)
    my_bar.progress(100)

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
    lr_model = LogisticRegression(random_state=1)
    lr_model.fit(X_train, y_train)
    #lr_pred = lr_model.predict(X_test_scaled)
    training_predictions = lr_model.predict(X_train)
    testing_predictions = lr_model.predict(X_test)

    lr_report_train = classification_report(y_train, training_predictions, output_dict=True)
    lr_report_train = pd.DataFrame(lr_report_train).transpose()

    lr_report_test = classification_report(y_test, testing_predictions, output_dict=True)
    lr_report_test = pd.DataFrame(lr_report_test).transpose()

    # lr_df = pd.DataFrame(index=data.index)
    # # lr_df['Predicted_signals'] = X_test
    # lr_df['Actual Returns'] = data["actual_returns"]
    # # lr_df['Strategy Returns'] = data["actual_returns"] * X_test
    # # st.write(lr_df)
    # lr_plot = lr_df.hvplot.line(title=f'How Logistic Regression is performing', ylabel='Price', xlabel='Date',height=500,
    #                             width=700)
    # st.session_state.lr_plot = lr_plot
    # st.write(lr_plot)
    
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

    # Using SVC (Model 4)
    training_begin = X.index.min()
    training_end = X.index.min() + DateOffset(months=3)
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]
    X_test = X.loc[training_end+DateOffset(hours=1):]
    y_test = y.loc[training_end+DateOffset(hours=1):]
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    svm_model = svm.SVC(kernel='rbf', probability=True, C=5,gamma=10)
    svm_model = svm_model.fit(X_train_scaled, y_train)
    
    svm_pred = svm_model.predict(X_test_scaled)
    svm_testing_report = classification_report(y_test, svm_pred, output_dict=True)
    svm_testing_report = pd.DataFrame(svm_testing_report).transpose()

    # # Using KNN 
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(X_train_scaled, y_train)
    # pred = knn.predict(X_test_scaled)

    # error_rate = []

    # for i in range(1,100):
    
    #     knn = KNeighborsClassifier(n_neighbors=5)
    #     knn.fit(X_train_scaled, y_train)
    #     pred_i = knn.predict(X_test_scaled)
    #     error_rate.append(np.mean(pred_i != y_test))
    
    # knn = KNeighborsClassifier(n_neighbors=1)
    # pred = knn.predict(X_test_scaled)
    
    # #Plot KNN
    # fig, ax = plt.subplots()
    # plt.figure(figsize=(10,6))
    # plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed', marker='o',
    #      markerfacecolor='red', markersize=10)
    # plt.title('Error Rate vs. K Value')
    # plt.xlabel('K')
    # plt.ylabel('Error Rate')
    # st.write(fig)
    
    
    # Using RandomForestClassifier (Model 5)
    rf_model = RandomForestClassifier()
    rfc_model = rf_model.fit(X_train_scaled, y_train)
    rf_pred = rfc_model.predict(X_test_scaled)
    rf_predictions_df = pd.DataFrame(index = X_test.index)
    rf_predictions_df['Predicted_signals'] = rf_pred
    data.reset_index(level=0, inplace=True)
    rf_predictions_df['actual_returns'] = data['actual_returns']
    rf_predictions_df["Strategy Returns"] = data['actual_returns']  * data['signal']
    rf_report = classification_report(y_test, rf_pred, output_dict=True)
    rf_report = pd.DataFrame(svm_testing_report).transpose()
    
    rf_plot = rf_predictions_df.hvplot.line(title=f'How RFC is performing', ylabel='Price', xlabel='Date',height=500,
                                width=700)
    st.session_state.rf_plot = rf_plot
    # fig, ax = plt.sublots()
    # plt(1 + rf_predictions_df[["Actual Returns", "Strategy Returns"]]).cumprod().plot(title="Baseline using RandomForest").get_figure()
    # st.write(fig)
    # rfc_plot = (1 + rf_predictions_df[["actual_returns", "Strategy Returns"]]).cumprod().hvplot(title="Baseline using KNN").get_figure()
    #.hvplot.line(title=f'How {user_choice} is performing', ylabel='Price', xlabel='Date',height=500,
    #                            width=700)
    # st.session_state.rfc_plot = rfc_plot
    # st.bokeh_chart(hv.render(rfc_plot))

    # Display report section
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Logistic Regression", "Gaussian", "Decision Tree", "SVM", "Random Forest"])

    with tab1:
        tab1.header("Logistic Regression Model")   
        cols = st.columns(2,gap='medium')

        with cols[0]:
            st.subheader("Training Results")
            st.write(lr_report_train, width=50)

        with cols[1]:
            st.subheader("Testing Results")
            st.write(lr_report_test, width=50)
        
        tab1.markdown("""---""")
        tab1.subheader("Summary")
        accuracy_lr = lr_report_test.loc["accuracy","f1-score"]
        recall_lr = lr_report_test.loc["macro avg","recall"]
        tab1.write(f"The overall accuracy of the Logistic Regression Model was {round(accuracy_lr,2)}")
        tab1.write(f"The overall recall of the Logistic Regression Model was {round(recall_lr,2)}")
    
    with tab2:
        tab2.header("GaussianNB Model")
        cols = st.columns(2,gap='medium')
        #tab2.image("https://static.streamlit.io/examples/dog.jpg", width=200)

        with cols[0]:
            st.subheader("Training Results")
            st.write(gauss_report_train)

        with cols[1]:
            st.subheader("Testing Results")
            st.write(gauss_report_test)
        
        tab2.markdown("""---""")
        tab2.subheader("Summary")
        accuracy_gauss = gauss_report_test.loc["accuracy","f1-score"]
        recall_gauss = gauss_report_test.loc["macro avg","recall"]
        tab2.write(f"The overall accuracy of the GaussianNB Model was {round(accuracy_gauss,2)}")
        tab2.write(f"The overall recall of the GaussianNB Model was {round(recall_gauss,2)}")
    
    with tab3:
        tab3.header("Decision Tree Classifier Model")
        cols = st.columns(2,gap='medium')
        #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

        with cols[0]:
            st.subheader("Training Results")
            st.write(tree_report_train)

        with cols[1]:
            st.subheader("Testing Results")
            st.write(tree_report_test)
        
        tab3.markdown("""---""")
        tab3.subheader("Summary")
        accuracy_tree = tree_report_test.loc["accuracy","f1-score"]
        recall_tree = tree_report_test.loc["macro avg","recall"]
        tab3.write(f"The overall accuracy of the Decision Tree Model was {round(accuracy_tree,2)}")
        tab3.write(f"The overall recall of the Decision Tree Model was {round(recall_tree,2)}")

    with tab4:
        tab4.header("SVM Model")
        cols = st.columns(1,gap='medium')  

        with cols[0]:
            st.subheader("Testing Results")
            st.write(svm_testing_report)
        
        tab4.markdown("""---""")
        tab4.subheader("Summary")
        accuracy_svm = svm_testing_report.loc["accuracy","f1-score"]
        tab4.write(f"The overall accuracy of the SVM Model was {round(accuracy_svm,2)}")

    with tab5:
        tab5.header("Random Forest Model")
        cols = st.columns(1,gap='medium')  

        with cols[0]:
            st.subheader("Testing Results")
            st.write(rf_report)
        
        tab5.markdown("""---""")
        tab5.subheader("Summary")
        accuracy_rf = rf_report.loc["f1-score", "accuracy"]
        st.bokeh_chart(hv.render(rf_plot))
        tab5.write(f"The overall accuracy of the SVM Model was {round(accuracy_rf,2)}")
    

