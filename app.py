import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date')
    return df

df = load_data()

def get_store_series(df, store_id):
    temp = df[df['Store'] == store_id].copy()
    temp = temp.sort_values('Date')
    temp.set_index('Date', inplace=True)
    temp = temp.asfreq('W-FRI')
    return temp['Weekly_Sales']

st.title("Walmart Sales Forecasting Dashboard")

stores = sorted(df['Store'].unique())
selected_store = st.selectbox("Select Store", stores)

series = get_store_series(df, selected_store)

st.line_chart(series)





















