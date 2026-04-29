import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

st.title("Walmart Time Series Forecasting Dashboard")

# -----------------------------
# STORE SELECTION
# -----------------------------
stores = [1, 20, 33]
store_choice = st.selectbox("Select Store", stores)

store_df = df[df['Store'] == store_choice].sort_values('Date')
store_df = store_df.set_index('Date')
series = store_df['Weekly_Sales']

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

horizon = len(test)

# =====================================================
# 1. ACTUAL TIME SERIES
# =====================================================
st.subheader("Actual Sales Data")

fig1, ax1 = plt.subplots()
ax1.plot(series.index, series, label="Actual Sales")
ax1.legend()
st.pyplot(fig1)

# =====================================================
# 2. HOLT-WINTERS MODEL
# =====================================================
hw_model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='add',
    seasonal_periods=52
)

hw_fit = hw_model.fit()
hw_forecast = hw_fit.forecast(horizon)

st.subheader("Holt-Winters Forecast")

fig2, ax2 = plt.subplots()
ax2.plot(train.index, train, label="Train")
ax2.plot(test.index, test, label="Actual")
ax2.plot(test.index, hw_forecast, label="Holt-Winters Forecast")
ax2.legend()
st.pyplot(fig2)

# =====================================================
# 3. SARIMA MODEL
# =====================================================
sarima_model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.forecast(horizon)

st.subheader("SARIMA Forecast")

fig3, ax3 = plt.subplots()
ax3.plot(train.index, train, label="Train")
ax3.plot(test.index, test, label="Actual")
ax3.plot(test.index, sarima_forecast, label="SARIMA Forecast")
ax3.legend()
st.pyplot(fig3)







