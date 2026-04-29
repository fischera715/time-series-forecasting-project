import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")
df['Date'] = pd.to_datetime(df['Date'])

stores = [1, 20, 33]

store_choice = st.selectbox("Select Store", stores)
horizon = st.slider("Forecast Horizon (weeks)", 4, 52, 12)

store_df = df[df['Store'] == store_choice].sort_values('Date')
store_df = store_df.set_index('Date')
series = store_df['Weekly_Sales']

model = ExponentialSmoothing(
    series,
    trend='add',
    seasonal='add',
    seasonal_periods=52
)

fit = model.fit()
forecast = fit.forecast(52)

st.title("Walmart Sales Forecasting")

st.subheader("Historical Sales")
st.line_chart(series)

st.subheader("Forecast (Holt-Winters)")
st.line_chart(forecast[:horizon])
