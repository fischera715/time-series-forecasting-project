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

def sarima_forecast(series, steps=52):
    model = SARIMAX(
        series,
        order=(1, 0, 1),
        seasonal_order=(0, 0, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)

    forecast_obj = results.get_forecast(steps=steps)
    pred = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    return results, pred, conf_int

st.subheader("SARIMA Forecast")

if st.button("Run SARIMA Forecast"):
    results, pred, conf_int = sarima_forecast(series)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(series, label="Actual")
    ax.plot(pred, label="Forecast")

    ax.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.3
    )

    ax.legend()
    st.pyplot(fig)

def holt_winters_forecast(series, steps=52):
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=52
    )

    fit = model.fit()
    forecast = fit.forecast(steps)

    return fit, forecast

st.subheader("Holt-Winters Forecast")

if st.button("Run Holt-Winters Forecast"):
    fit, forecast = holt_winters_forecast(series)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(series, label="Actual")
    ax.plot(forecast, label="Forecast")

    ax.legend()
    st.pyplot(fig)





















